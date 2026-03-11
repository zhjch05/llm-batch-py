use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBool, PyBytes, PyDict, PyFloat, PyInt, PyList, PyString, PyTuple};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;

fn canonical_from_py(value: &Bound<'_, PyAny>) -> PyResult<String> {
    if value.is_none() {
        return Ok("null".to_string());
    }
    if value.is_instance_of::<PyBool>() {
        let flag = value.extract::<bool>()?;
        return Ok(if flag { "true" } else { "false" }.to_string());
    }
    if value.is_instance_of::<PyInt>() {
        return Ok(value.str()?.to_str()?.to_string());
    }
    if value.is_instance_of::<PyFloat>() {
        let number = value.extract::<f64>()?;
        if !number.is_finite() {
            return Err(PyTypeError::new_err("non-finite floats are not supported"));
        }
        let mut text = serde_json::to_string(&number).unwrap();
        if text.ends_with(".0") {
            text.truncate(text.len() - 2);
        }
        return Ok(text);
    }
    if value.is_instance_of::<PyString>() {
        return Ok(serde_json::to_string(&value.extract::<String>()?).unwrap());
    }
    if value.is_instance_of::<PyBytes>() {
        let text = std::str::from_utf8(value.extract::<&[u8]>()?)
            .map_err(|err| PyTypeError::new_err(err.to_string()))?;
        return Ok(serde_json::to_string(text).unwrap());
    }
    if let Ok(dict) = value.downcast::<PyDict>() {
        let mut entries: Vec<(String, String)> = Vec::new();
        for (key, entry_value) in dict.iter() {
            let key_text = if key.is_instance_of::<PyString>() {
                key.extract::<String>()?
            } else {
                key.str()?.to_str()?.to_string()
            };
            entries.push((key_text, canonical_from_py(&entry_value)?));
        }
        entries.sort_by(|left, right| left.0.cmp(&right.0));
        let body = entries
            .into_iter()
            .map(|(key, rendered)| format!("{}:{}", serde_json::to_string(&key).unwrap(), rendered))
            .collect::<Vec<_>>()
            .join(",");
        return Ok(format!("{{{body}}}"));
    }
    if let Ok(list) = value.downcast::<PyList>() {
        let body = list
            .iter()
            .map(|item| canonical_from_py(&item))
            .collect::<PyResult<Vec<_>>>()?
            .join(",");
        return Ok(format!("[{body}]"));
    }
    if let Ok(tuple) = value.downcast::<PyTuple>() {
        let body = tuple
            .iter()
            .map(|item| canonical_from_py(&item))
            .collect::<PyResult<Vec<_>>>()?
            .join(",");
        return Ok(format!("[{body}]"));
    }
    if value.hasattr("model_dump")? {
        let kwargs = PyDict::new(value.py());
        kwargs.set_item("mode", "json")?;
        let dumped = value.getattr("model_dump")?.call((), Some(&kwargs))?;
        return canonical_from_py(&dumped);
    }
    if value.hasattr("isoformat")? {
        match value.call_method0("isoformat") {
            Ok(rendered) => return canonical_from_py(&rendered),
            Err(err) if err.is_instance_of::<PyTypeError>(value.py()) => {
                return Ok(serde_json::to_string(&value.str()?.to_str()?).unwrap());
            }
            Err(err) => return Err(err),
        }
    }
    Err(PyTypeError::new_err(format!(
        "unsupported type for canonical serialization: {}",
        value.get_type().name()?
    )))
}

fn row_to_dict<'py>(row: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyDict>> {
    if let Ok(dict) = row.downcast::<PyDict>() {
        return Ok(dict.clone());
    }
    let dict_value = row.py().import("builtins")?.getattr("dict")?.call1((row,))?;
    dict_value
        .downcast::<PyDict>()
        .map(|dict| dict.clone())
        .map_err(|_| PyTypeError::new_err("template rows must be mappings"))
}

fn row_value_to_text(value: &Bound<'_, PyAny>) -> PyResult<String> {
    if value.is_none() {
        return Ok(String::new());
    }
    if value.is_instance_of::<PyString>() {
        return value.extract::<String>();
    }
    Ok(value.str()?.to_str()?.to_string())
}

#[derive(Default)]
struct SnapshotConfig {
    include: Vec<String>,
    exclude: Vec<String>,
    priority: Vec<String>,
}

fn parse_string_list(py: Python<'_>, input: &str) -> PyResult<Vec<String>> {
    let parsed = py.import("ast")?.getattr("literal_eval")?.call1((input,))?;
    let list = parsed.downcast::<PyList>().map_err(|_| {
        PyValueError::new_err("row_snapshot values must be list[str]")
    })?;
    let mut items = Vec::with_capacity(list.len());
    for item in list.iter() {
        items.push(item.extract::<String>().map_err(|_| {
            PyValueError::new_err("row_snapshot values must be list[str]")
        })?);
    }
    Ok(items)
}

fn parse_snapshot_args(py: Python<'_>, args: &str) -> PyResult<SnapshotConfig> {
    let mut config = SnapshotConfig::default();
    if args.trim().is_empty() {
        return Ok(config);
    }

    let bytes = args.as_bytes();
    let mut index = 0usize;
    while index < bytes.len() {
        while index < bytes.len() && (bytes[index].is_ascii_whitespace() || bytes[index] == b',') {
            index += 1;
        }
        if index >= bytes.len() {
            break;
        }

        let start = index;
        while index < bytes.len() && bytes[index].is_ascii_alphabetic() {
            index += 1;
        }
        let key = args[start..index].trim();
        while index < bytes.len() && bytes[index].is_ascii_whitespace() {
            index += 1;
        }
        if index >= bytes.len() || bytes[index] != b'=' {
            return Err(PyValueError::new_err(
                "row_snapshot arguments must use key=value",
            ));
        }
        index += 1;
        while index < bytes.len() && bytes[index].is_ascii_whitespace() {
            index += 1;
        }
        if index >= bytes.len() || bytes[index] != b'[' {
            return Err(PyValueError::new_err(
                "row_snapshot arguments must be list[str]",
            ));
        }

        let list_start = index;
        let mut depth = 0usize;
        let mut quote: Option<u8> = None;
        while index < bytes.len() {
            let current = bytes[index];
            if let Some(active_quote) = quote {
                if current == b'\\' {
                    index += 2;
                    continue;
                }
                if current == active_quote {
                    quote = None;
                }
                index += 1;
                continue;
            }
            if current == b'\'' || current == b'"' {
                quote = Some(current);
                index += 1;
                continue;
            }
            if current == b'[' {
                depth += 1;
            } else if current == b']' {
                depth -= 1;
                if depth == 0 {
                    index += 1;
                    break;
                }
            }
            index += 1;
        }

        let parsed = parse_string_list(py, &args[list_start..index])?;
        match key {
            "include" => config.include = parsed,
            "exclude" => config.exclude = parsed,
            "priority" => config.priority = parsed,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "Unsupported row_snapshot option: {key}"
                )))
            }
        }
    }

    Ok(config)
}

fn split_filters(expr: &str) -> (String, Vec<String>) {
    let segments = expr
        .split('|')
        .map(|segment| segment.trim().to_string())
        .collect::<Vec<_>>();
    let base = segments.first().cloned().unwrap_or_default();
    let filters = segments
        .into_iter()
        .skip(1)
        .filter(|segment| !segment.is_empty())
        .collect();
    (base, filters)
}

fn apply_filters(mut rendered: String, filters: &[String]) -> PyResult<String> {
    for filter_name in filters {
        match filter_name.as_str() {
            "upper" => rendered = rendered.to_uppercase(),
            "lower" => rendered = rendered.to_lowercase(),
            _ => {
                return Err(PyValueError::new_err(format!(
                    "Unsupported template filter: {filter_name}"
                )))
            }
        }
    }
    Ok(rendered)
}

fn render_row_snapshot(row: &Bound<'_, PyAny>, args: &str) -> PyResult<String> {
    let py = row.py();
    let config = parse_snapshot_args(py, args)?;
    let dict = row_to_dict(row)?;
    let exclude: BTreeSet<String> = config.exclude.into_iter().collect();
    let mut selected: Vec<(String, Bound<'_, PyAny>)> = Vec::new();

    if config.include.is_empty() {
        for (key, value) in dict.iter() {
            let key_text = if key.is_instance_of::<PyString>() {
                key.extract::<String>()?
            } else {
                key.str()?.to_str()?.to_string()
            };
            if !exclude.contains(&key_text) {
                selected.push((key_text, value));
            }
        }
    } else {
        for key in config.include {
            if exclude.contains(&key) {
                continue;
            }
            if let Some(value) = dict.get_item(&key)? {
                selected.push((key, value));
            }
        }
    }

    selected.sort_by(|left, right| left.0.cmp(&right.0));
    let mut ordered: Vec<(String, Bound<'_, PyAny>)> = Vec::new();
    for priority_key in config.priority {
        if let Some(index) = selected.iter().position(|(key, _)| key == &priority_key) {
            ordered.push(selected.remove(index));
        }
    }
    ordered.extend(selected);

    let body = ordered
        .into_iter()
        .map(|(key, value)| {
            Ok(format!(
                "{}:{}",
                serde_json::to_string(&key).unwrap(),
                canonical_from_py(&value)?
            ))
        })
        .collect::<PyResult<Vec<_>>>()?
        .join(",");
    Ok(format!("{{{body}}}"))
}

fn evaluate_expr<'py>(
    py: Python<'py>,
    expr: &str,
    row: &Bound<'py, PyAny>,
) -> PyResult<Py<PyAny>> {
    let (base_expr, filters) = split_filters(expr);
    let base = base_expr.trim();
    let value = if let Some(field_name) = base.strip_prefix("row.") {
        let dict = row_to_dict(row)?;
        match dict.get_item(field_name)? {
            Some(value) => value.unbind(),
            None => {
                return Err(PyValueError::new_err(format!(
                    "Missing row field referenced in template: {field_name}"
                )))
            }
        }
    } else if let Some(args) = base
        .strip_prefix("row_snapshot(")
        .and_then(|value| value.strip_suffix(')'))
    {
        PyString::new(py, &render_row_snapshot(row, args)?)
            .into_any()
            .unbind()
    } else {
        return Err(PyValueError::new_err(format!(
            "Unsupported template expression: {base}"
        )));
    };

    if filters.is_empty() {
        return Ok(value);
    }
    let rendered = apply_filters(row_value_to_text(value.bind(py))?, &filters)?;
    Ok(PyString::new(py, &rendered).into_any().unbind())
}

fn render_expr(expr: &str, row: &Bound<'_, PyAny>) -> PyResult<String> {
    row_value_to_text(evaluate_expr(row.py(), expr, row)?.bind(row.py()))
}

fn collect_expr_fields(py: Python<'_>, expr: &str, columns: &mut BTreeSet<String>) -> PyResult<()> {
    let (base_expr, _filters) = split_filters(expr);
    let base = base_expr.trim();
    if let Some(field_name) = base.strip_prefix("row.") {
        columns.insert(field_name.to_string());
        return Ok(());
    }
    if let Some(args) = base
        .strip_prefix("row_snapshot(")
        .and_then(|value| value.strip_suffix(')'))
    {
        let config = parse_snapshot_args(py, args)?;
        let exclude: BTreeSet<String> = config.exclude.into_iter().collect();
        columns.extend(config.include.into_iter().filter(|key| !exclude.contains(key)));
        return Ok(());
    }
    Err(PyValueError::new_err(format!(
        "Unsupported template expression: {base}"
    )))
}

fn placeholder_spans(template: &str) -> PyResult<Vec<(usize, usize, String)>> {
    let mut spans = Vec::new();
    let mut cursor = 0usize;
    while let Some(open_rel) = template[cursor..].find("{{") {
        let open = cursor + open_rel;
        let close_rel = template[open + 2..]
            .find("}}")
            .ok_or_else(|| PyValueError::new_err("Unclosed template placeholder"))?;
        let close = open + 2 + close_rel;
        spans.push((open, close + 2, template[open + 2..close].trim().to_string()));
        cursor = close + 2;
    }
    Ok(spans)
}

#[pyfunction]
fn canonical_json(value: &Bound<'_, PyAny>) -> PyResult<String> {
    canonical_from_py(value)
}

#[pyfunction]
fn stable_hash(value: &Bound<'_, PyAny>) -> PyResult<String> {
    let canonical = canonical_from_py(value)?;
    let mut hasher = Sha256::new();
    hasher.update(canonical.as_bytes());
    Ok(format!("{:x}", hasher.finalize()))
}

#[pyfunction]
fn jsonl_dump_bytes(values: &Bound<'_, PyAny>) -> PyResult<Vec<u8>> {
    let rendered = values
        .try_iter()?
        .map(|item| canonical_from_py(&item?))
        .collect::<PyResult<Vec<_>>>()?
        .join("\n");
    Ok(format!("{rendered}\n").into_bytes())
}

#[pyfunction]
fn render_template_string(template: &str, row: &Bound<'_, PyAny>) -> PyResult<String> {
    let spans = placeholder_spans(template)?;
    let mut rendered = String::new();
    let mut cursor = 0usize;
    for (start, end, expr) in spans {
        rendered.push_str(&template[cursor..start]);
        rendered.push_str(&render_expr(&expr, row)?);
        cursor = end;
    }
    rendered.push_str(&template[cursor..]);
    Ok(rendered)
}

#[pyfunction]
fn evaluate_template_expr(
    py: Python<'_>,
    expr: &str,
    row: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    evaluate_expr(py, expr, row)
}

#[pyfunction]
fn extract_template_fields(py: Python<'_>, template: &str) -> PyResult<Vec<String>> {
    let mut columns = BTreeSet::new();
    for (_, _, expr) in placeholder_spans(template)? {
        collect_expr_fields(py, &expr, &mut columns)?;
    }
    Ok(columns.into_iter().collect())
}

#[pymodule]
fn _core(_py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(canonical_json, module)?)?;
    module.add_function(wrap_pyfunction!(stable_hash, module)?)?;
    module.add_function(wrap_pyfunction!(jsonl_dump_bytes, module)?)?;
    module.add_function(wrap_pyfunction!(render_template_string, module)?)?;
    module.add_function(wrap_pyfunction!(evaluate_template_expr, module)?)?;
    module.add_function(wrap_pyfunction!(extract_template_fields, module)?)?;
    Ok(())
}
