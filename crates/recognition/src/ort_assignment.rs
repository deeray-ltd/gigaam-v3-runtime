// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 Yuriy Krasilnikov
// Copyright (C) 2026 Deeray Ltd.
//
// This file is part of GigaAM v3 Runtime, free software distributed under
// the terms of the GNU Lesser General Public License, either version 3 of the
// License, or (at your option) any later version. See COPYING.LESSER and COPYING
// for the full terms. There is NO WARRANTY, to the extent permitted by law.

//! API-24 ONNX Runtime graph-assignment inspection and canonical CUDA evidence.

use crate::provider::CudaAssignmentFingerprint;
use ort::AsPointer;
use ort::session::Session;
use sha2::{Digest, Sha256};
use std::ffi::CStr;
use std::slice;

const MAX_ASSIGNMENT_RECORDS: usize = 1_000_000;
const FINGERPRINT_DOMAIN: &[u8] = b"gigaam-v3-runtime/cuda-assignment/v1";

/// Owned, read-only evidence from one CUDA encoder session's observed graph assignment.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CudaAssignmentEvidence {
    fingerprint: CudaAssignmentFingerprint,
    cpu_assignments: usize,
    cuda_assignments: usize,
}

impl CudaAssignmentEvidence {
    pub const fn fingerprint(&self) -> &CudaAssignmentFingerprint {
        &self.fingerprint
    }

    pub const fn cpu_assignments(&self) -> usize {
        self.cpu_assignments
    }

    pub const fn cuda_assignments(&self) -> usize {
        self.cuda_assignments
    }
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct AssignmentRecord {
    provider: String,
    node: String,
    domain: String,
    operator_type: String,
}

pub(crate) fn inspect(session: &Session) -> Result<CudaAssignmentEvidence, String> {
    evidence_from_records(assignment_records(session)?)
}

/// Validates and binds copied CUDA graph-assignment records into owned evidence.
fn evidence_from_records(
    mut records: Vec<AssignmentRecord>,
) -> Result<CudaAssignmentEvidence, String> {
    if records.is_empty() {
        return Err(
            "ONNX Runtime reported no graph assignment records for a CUDA encoder session".into(),
        );
    }
    let mut cpu_assignments = 0_usize;
    let mut cuda_assignments = 0_usize;
    for record in &records {
        match record.provider.as_str() {
            "CPUExecutionProvider" => {
                cpu_assignments = cpu_assignments
                    .checked_add(1)
                    .ok_or_else(|| "CPU assignment count overflows usize".to_owned())?;
            }
            "CUDAExecutionProvider" => {
                cuda_assignments = cuda_assignments
                    .checked_add(1)
                    .ok_or_else(|| "CUDA assignment count overflows usize".to_owned())?;
            }
            provider => {
                return Err(format!(
                    "CUDA encoder graph assigned node {:?} to unsupported provider {provider:?}",
                    record.node
                ));
            }
        }
    }
    if cuda_assignments == 0 {
        return Err("CUDA encoder graph assignment contains no CUDAExecutionProvider nodes".into());
    }
    records.sort_unstable();
    Ok(CudaAssignmentEvidence {
        fingerprint: fingerprint(&records)?,
        cpu_assignments,
        cuda_assignments,
    })
}

/// Reads assignment metadata while `session` remains borrowed and alive.
///
/// # Safety proof
///
/// ONNX Runtime v1.29 documents these assignment objects and C strings as runtime-owned borrows
/// of the live session. This function retains its `&Session` borrow through every foreign call,
/// checks each status, pointer, and length before use, copies each string immediately into Rust
/// ownership, never stores a foreign pointer, and never frees assignment objects or strings.
/// Each returned status is transferred once to `ort::Error`, which owns its release.
fn assignment_records(session: &Session) -> Result<Vec<AssignmentRecord>, String> {
    let api = ort::api();
    let mut raw_subgraphs: *const *const ort::sys::OrtEpAssignedSubgraph = std::ptr::null();
    let mut subgraph_count = 0_usize;

    // SAFETY: `session.ptr()` is non-null by `AsPointer`; both out-pointers remain valid for this call.
    let status = unsafe {
        (api.Session_GetEpGraphAssignmentInfo)(
            session.ptr(),
            &mut raw_subgraphs,
            &mut subgraph_count,
        )
    };
    // SAFETY: the preceding ONNX Runtime API call returned this status and transfers its ownership here.
    unsafe { ort::Error::result_from_status(status) }
        .map_err(|error| format!("read ONNX Runtime graph assignment information: {error}"))?;

    let maximum_slice_length = usize::try_from(isize::MAX)
        .map_err(|_| "platform cannot represent the maximum FFI slice length".to_owned())?;
    if subgraph_count > MAX_ASSIGNMENT_RECORDS || subgraph_count > maximum_slice_length {
        return Err("ONNX Runtime graph assignment subgraph count exceeds the safety bound".into());
    }
    if subgraph_count > 0 && raw_subgraphs.is_null() {
        return Err("ONNX Runtime returned a null graph-assignment subgraph array".into());
    }
    let subgraphs: &[*const ort::sys::OrtEpAssignedSubgraph] = if subgraph_count == 0 {
        &[]
    } else {
        // SAFETY: ONNX Runtime returned a non-null array with a checked length for this live session.
        unsafe { slice::from_raw_parts(raw_subgraphs, subgraph_count) }
    };

    let mut records = Vec::new();
    for &subgraph in subgraphs {
        if subgraph.is_null() {
            return Err("ONNX Runtime returned a null graph-assignment subgraph".into());
        }

        let mut provider_pointer = std::ptr::null();
        // SAFETY: `subgraph` is non-null and borrowed from the live session; the output pointer lives through the call.
        let status = unsafe { (api.EpAssignedSubgraph_GetEpName)(subgraph, &mut provider_pointer) };
        // SAFETY: the preceding ONNX Runtime API call returned this status and transfers its ownership here.
        unsafe { ort::Error::result_from_status(status) }
            .map_err(|error| format!("read ONNX Runtime assignment provider: {error}"))?;
        if provider_pointer.is_null() {
            return Err("ONNX Runtime returned a null graph-assignment provider name".into());
        }
        // SAFETY: ONNX Runtime returned a non-null NUL-terminated borrowed string; copy it before another FFI call.
        let provider_bytes = unsafe { CStr::from_ptr(provider_pointer).to_bytes().to_vec() };
        let provider = String::from_utf8(provider_bytes).map_err(|_| {
            "ONNX Runtime returned a non-UTF-8 graph-assignment provider name".to_owned()
        })?;

        let mut raw_nodes: *const *const ort::sys::OrtEpAssignedNode = std::ptr::null();
        let mut node_count = 0_usize;
        // SAFETY: `subgraph` is non-null and borrowed from the live session; both out-pointers remain valid for this call.
        let status =
            unsafe { (api.EpAssignedSubgraph_GetNodes)(subgraph, &mut raw_nodes, &mut node_count) };
        // SAFETY: the preceding ONNX Runtime API call returned this status and transfers its ownership here.
        unsafe { ort::Error::result_from_status(status) }
            .map_err(|error| format!("read ONNX Runtime graph-assignment nodes: {error}"))?;
        if node_count > maximum_slice_length {
            return Err(
                "ONNX Runtime graph assignment node count exceeds the platform slice bound".into(),
            );
        }
        if node_count > 0 && raw_nodes.is_null() {
            return Err("ONNX Runtime returned a null graph-assignment node array".into());
        }
        let next_record_count = records.len().checked_add(node_count).ok_or_else(|| {
            "ONNX Runtime graph assignment record count overflows usize".to_owned()
        })?;
        if next_record_count > MAX_ASSIGNMENT_RECORDS {
            return Err(
                "ONNX Runtime graph assignment record count exceeds the safety bound".into(),
            );
        }
        let nodes: &[*const ort::sys::OrtEpAssignedNode] = if node_count == 0 {
            &[]
        } else {
            // SAFETY: ONNX Runtime returned a non-null array with a checked length for this live session.
            unsafe { slice::from_raw_parts(raw_nodes, node_count) }
        };

        for &node in nodes {
            if node.is_null() {
                return Err("ONNX Runtime returned a null graph-assignment node".into());
            }
            let node_name = assignment_node_string(api, node, AssignmentNodeField::Name)?;
            let domain = assignment_node_string(api, node, AssignmentNodeField::Domain)?;
            let operator_type =
                assignment_node_string(api, node, AssignmentNodeField::OperatorType)?;
            records.push(AssignmentRecord {
                provider: provider.clone(),
                node: node_name,
                domain,
                operator_type,
            });
        }
    }
    Ok(records)
}

#[derive(Clone, Copy)]
enum AssignmentNodeField {
    Name,
    Domain,
    OperatorType,
}

impl AssignmentNodeField {
    const fn description(self) -> &'static str {
        match self {
            Self::Name => "name",
            Self::Domain => "domain",
            Self::OperatorType => "operator type",
        }
    }
}

fn assignment_node_string(
    api: &ort::sys::OrtApi,
    node: *const ort::sys::OrtEpAssignedNode,
    field: AssignmentNodeField,
) -> Result<String, String> {
    let mut pointer = std::ptr::null();
    let status = match field {
        AssignmentNodeField::Name => {
            // SAFETY: `node` was checked non-null and is borrowed from the live session; the output pointer lives through the call.
            unsafe { (api.EpAssignedNode_GetName)(node, &mut pointer) }
        }
        AssignmentNodeField::Domain => {
            // SAFETY: `node` was checked non-null and is borrowed from the live session; the output pointer lives through the call.
            unsafe { (api.EpAssignedNode_GetDomain)(node, &mut pointer) }
        }
        AssignmentNodeField::OperatorType => {
            // SAFETY: `node` was checked non-null and is borrowed from the live session; the output pointer lives through the call.
            unsafe { (api.EpAssignedNode_GetOperatorType)(node, &mut pointer) }
        }
    };
    // SAFETY: the immediately preceding ONNX Runtime API call returned this status and transfers its ownership here.
    unsafe { ort::Error::result_from_status(status) }.map_err(|error| {
        format!(
            "read ONNX Runtime assignment node {}: {error}",
            field.description()
        )
    })?;
    if pointer.is_null() {
        return Err(format!(
            "ONNX Runtime returned a null graph-assignment node {}",
            field.description()
        ));
    }
    // SAFETY: ONNX Runtime returned a non-null NUL-terminated borrowed string; copy it before returning.
    let bytes = unsafe { CStr::from_ptr(pointer).to_bytes().to_vec() };
    String::from_utf8(bytes).map_err(|_| {
        format!(
            "ONNX Runtime returned a non-UTF-8 graph-assignment node {}",
            field.description()
        )
    })
}

fn fingerprint(records: &[AssignmentRecord]) -> Result<CudaAssignmentFingerprint, String> {
    let total_count = u64::try_from(records.len())
        .map_err(|_| "graph assignment record count exceeds u64".to_owned())?;
    let mut hasher = Sha256::new();
    write_bytes(&mut hasher, FINGERPRINT_DOMAIN)?;
    write_u64(&mut hasher, total_count);

    let mut position = 0_usize;
    while position < records.len() {
        let record = records
            .get(position)
            .ok_or_else(|| "graph assignment record position is unavailable".to_owned())?;
        let mut end = position
            .checked_add(1)
            .ok_or_else(|| "graph assignment record range overflows usize".to_owned())?;
        while records.get(end) == Some(record) {
            end = end
                .checked_add(1)
                .ok_or_else(|| "graph assignment record range overflows usize".to_owned())?;
        }
        let multiplicity = end
            .checked_sub(position)
            .and_then(|count| u64::try_from(count).ok())
            .ok_or_else(|| "graph assignment record multiplicity exceeds u64".to_owned())?;
        write_u64(&mut hasher, multiplicity);
        write_bytes(&mut hasher, record.provider.as_bytes())?;
        write_bytes(&mut hasher, record.node.as_bytes())?;
        write_bytes(&mut hasher, record.domain.as_bytes())?;
        write_bytes(&mut hasher, record.operator_type.as_bytes())?;
        position = end;
    }
    let digest: [u8; 32] = hasher.finalize().into();
    Ok(CudaAssignmentFingerprint::from_bytes(digest))
}

fn write_u64(hasher: &mut Sha256, value: u64) {
    hasher.update(value.to_be_bytes());
}

fn write_bytes(hasher: &mut Sha256, bytes: &[u8]) -> Result<(), String> {
    let length = u64::try_from(bytes.len())
        .map_err(|_| "canonical graph assignment field length exceeds u64".to_owned())?;
    write_u64(hasher, length);
    hasher.update(bytes);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{AssignmentRecord, evidence_from_records};

    fn record(provider: &str, node: &str, domain: &str, operator_type: &str) -> AssignmentRecord {
        AssignmentRecord {
            provider: provider.into(),
            node: node.into(),
            domain: domain.into(),
            operator_type: operator_type.into(),
        }
    }

    #[test]
    fn evidence_is_order_independent_and_sensitive_to_every_assignment_field() -> Result<(), String>
    {
        let cuda = record("CUDAExecutionProvider", "encoder/conv", "", "Conv");
        let cpu = record("CPUExecutionProvider", "encoder/cast", "ai.onnx", "Cast");
        let records = vec![cuda.clone(), cpu.clone(), cuda.clone()];
        let evidence = evidence_from_records(records.clone())?;
        assert_eq!(evidence.cpu_assignments(), 1);
        assert_eq!(evidence.cuda_assignments(), 2);

        let reordered = evidence_from_records(vec![cpu.clone(), cuda.clone(), cuda.clone()])?;
        assert_eq!(evidence.fingerprint(), reordered.fingerprint());

        let provider = evidence_from_records(vec![
            cuda.clone(),
            record("CUDAExecutionProvider", "encoder/cast", "ai.onnx", "Cast"),
            cuda.clone(),
        ])?;
        assert_ne!(evidence.fingerprint(), provider.fingerprint());

        let node = evidence_from_records(vec![
            record("CUDAExecutionProvider", "encoder/conv-2", "", "Conv"),
            cpu.clone(),
            cuda.clone(),
        ])?;
        assert_ne!(evidence.fingerprint(), node.fingerprint());

        let domain = evidence_from_records(vec![
            record(
                "CUDAExecutionProvider",
                "encoder/conv",
                "com.example",
                "Conv",
            ),
            cpu.clone(),
            cuda.clone(),
        ])?;
        assert_ne!(evidence.fingerprint(), domain.fingerprint());

        let operator_type = evidence_from_records(vec![
            record("CUDAExecutionProvider", "encoder/conv", "", "Relu"),
            cpu.clone(),
            cuda.clone(),
        ])?;
        assert_ne!(evidence.fingerprint(), operator_type.fingerprint());

        let without_duplicate = evidence_from_records(vec![cuda, cpu])?;
        assert_ne!(evidence.fingerprint(), without_duplicate.fingerprint());
        Ok(())
    }

    #[test]
    fn evidence_refuses_empty_unknown_and_cpu_only_assignments() {
        assert!(evidence_from_records(Vec::new()).is_err());
        assert!(
            evidence_from_records(vec![record(
                "TensorRTExecutionProvider",
                "encoder/node",
                "",
                "Conv",
            )])
            .is_err()
        );
        assert!(
            evidence_from_records(vec![record(
                "CPUExecutionProvider",
                "encoder/node",
                "",
                "Conv",
            )])
            .is_err()
        );
    }
}
