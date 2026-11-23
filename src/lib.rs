use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use optimal1d::microagg::{microaggregation, CostKind, Method};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Optimal microaggregation of an already **sorted** 1d float array.
#[pyfunction]
#[pyo3(signature = (x, k, method="staggered", cost="sse", stable=1))]
fn microaggregation_sorted<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    k: usize,
    method: &str,
    cost: &str,
    stable: u8,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let method = Method::parse(method).ok_or_else(|| PyValueError::new_err("invalid method supplied"))?;
    let cost = CostKind::parse(cost).ok_or_else(|| PyValueError::new_err("invalid cost supplied"))?;
    let x = x.as_slice().map_err(|_| PyValueError::new_err("x must be contiguous"))?;
    let out = py
        .detach(|| microaggregation(x, k, method, cost, stable))
        .map_err(PyValueError::new_err)?;
    Ok(out.into_pyarray(py))
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(microaggregation_sorted, m)?)?;
    Ok(())
}
