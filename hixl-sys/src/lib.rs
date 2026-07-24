// FFI bindings to the production HiXL C shim built by this crate.
//
// The shim wraps the HiXL C++ API behind a flat C interface,
// mirroring the role that rdmaxcel-sys plays for ibverbs.
//
// All HiXL calls are made directly on the caller's thread after
// restoring the ACL context via aclrtSetCurrentContext.  No internal
// worker thread serialisation — callers may invoke from any thread.

#![allow(non_camel_case_types)]

#[cfg(not(feature = "mock"))]
use std::os::raw::c_char;
#[cfg(not(feature = "mock"))]
use std::os::raw::c_int;
use std::os::raw::c_void;

#[cfg(not(feature = "mock"))]
unsafe extern "C" {
    /// Initialize a HiXL engine on the given NPU device.
    /// Returns an opaque context pointer, or null on failure.
    pub fn hixl_init_engine(dev: c_int, engine_id: *const c_char) -> *mut c_void;

    /// Establish a connection to a remote engine.
    /// Returns 0 on success.
    pub fn hixl_connect(
        ctx: *mut c_void,
        remote_engine_id: *const c_char,
        timeout_ms: c_int,
    ) -> c_int;

    /// Register a device memory region for RDMA access.
    /// Returns 0 on success.
    pub fn hixl_register_mem(ctx: *mut c_void, addr: usize, size: usize) -> c_int;

    /// Deregister a previously registered memory region.
    /// Returns 0 on success, -1 if the address was not registered.
    pub fn hixl_deregister_mem(ctx: *mut c_void, addr: usize) -> c_int;

    /// Single-sided READ: pull remote data into local buffer.
    /// Returns 0 on success.
    pub fn hixl_transfer_read(
        ctx: *mut c_void,
        remote_engine_id: *const c_char,
        local_addr: usize,
        remote_addr: usize,
        len: usize,
        timeout_ms: c_int,
    ) -> c_int;

    /// Single-sided WRITE: push local data into remote buffer.
    /// Returns 0 on success.
    pub fn hixl_transfer_write(
        ctx: *mut c_void,
        remote_engine_id: *const c_char,
        local_addr: usize,
        remote_addr: usize,
        len: usize,
        timeout_ms: c_int,
    ) -> c_int;

    /// Return the ACL context saved during engine init.
    pub fn hixl_get_acl_context(ctx: *mut c_void) -> usize;

    /// Probe whether HCCS IPC memory export is supported on the given device.
    /// Returns 0 if supported, non-zero otherwise.
    /// Does NOT require an existing HixlEngine — can be called before init.
    pub fn hixl_probe_hccs(dev: c_int) -> c_int;

    /// Finalize and release engine resources.
    pub fn hixl_cleanup(ctx: *mut c_void);
}

/// Wrapper around the raw HiXL context pointer.
/// Send + Sync because the C shim restores ACL context before each
/// operation, making it safe to call from any thread.
#[cfg(not(feature = "mock"))]
#[derive(Debug)]
pub struct HixlEngine {
    ptr: *mut c_void,
    dev: i32,
    acl_ctx: usize,
}

#[cfg(not(feature = "mock"))]
unsafe impl Send for HixlEngine {}
#[cfg(not(feature = "mock"))]
unsafe impl Sync for HixlEngine {}

#[cfg(not(feature = "mock"))]
impl HixlEngine {
    pub fn new(dev: i32, engine_id: &str) -> Result<Self, String> {
        let c_eid =
            std::ffi::CString::new(engine_id).map_err(|e| format!("invalid engine_id: {e}"))?;
        let ptr = unsafe { hixl_init_engine(dev, c_eid.as_ptr()) };
        if ptr.is_null() {
            Err(format!(
                "hixl_init_engine failed for dev={dev} engine_id={engine_id}"
            ))
        } else {
            let acl_ctx = unsafe { hixl_get_acl_context(ptr) };
            Ok(Self { ptr, dev, acl_ctx })
        }
    }

    pub fn ptr(&self) -> *mut c_void {
        self.ptr
    }

    pub fn dev(&self) -> i32 {
        self.dev
    }

    pub fn acl_ctx(&self) -> usize {
        self.acl_ctx
    }

    pub fn connect(&self, remote_engine_id: &str, timeout_ms: i32) -> Result<(), i32> {
        let c_eid = std::ffi::CString::new(remote_engine_id).unwrap();
        let ret = unsafe { hixl_connect(self.ptr, c_eid.as_ptr(), timeout_ms) };
        if ret == 0 { Ok(()) } else { Err(ret) }
    }

    pub fn register_mem(&self, addr: usize, size: usize) -> Result<(), i32> {
        let ret = unsafe { hixl_register_mem(self.ptr, addr, size) };
        if ret == 0 { Ok(()) } else { Err(ret) }
    }

    pub fn deregister_mem(&self, addr: usize) -> Result<(), i32> {
        let ret = unsafe { hixl_deregister_mem(self.ptr, addr) };
        if ret == 0 { Ok(()) } else { Err(ret) }
    }

    pub fn transfer_read(
        &self,
        remote_engine_id: &str,
        local_addr: usize,
        remote_addr: usize,
        len: usize,
        timeout_ms: i32,
    ) -> Result<(), i32> {
        let c_eid = std::ffi::CString::new(remote_engine_id).unwrap();
        let ret = unsafe {
            hixl_transfer_read(
                self.ptr,
                c_eid.as_ptr(),
                local_addr,
                remote_addr,
                len,
                timeout_ms,
            )
        };
        if ret == 0 { Ok(()) } else { Err(ret) }
    }

    pub fn transfer_write(
        &self,
        remote_engine_id: &str,
        local_addr: usize,
        remote_addr: usize,
        len: usize,
        timeout_ms: i32,
    ) -> Result<(), i32> {
        let c_eid = std::ffi::CString::new(remote_engine_id).unwrap();
        let ret = unsafe {
            hixl_transfer_write(
                self.ptr,
                c_eid.as_ptr(),
                local_addr,
                remote_addr,
                len,
                timeout_ms,
            )
        };
        if ret == 0 { Ok(()) } else { Err(ret) }
    }
}

#[cfg(not(feature = "mock"))]
impl Drop for HixlEngine {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { hixl_cleanup(self.ptr) };
        }
    }
}

/// Probe whether HCCS IPC memory export works on the given device.
/// Returns `true` if HCCS is available, `false` otherwise.
#[cfg(not(feature = "mock"))]
pub fn probe_hccs(dev: i32) -> bool {
    let ret = unsafe { hixl_probe_hccs(dev) };
    ret == 0
}

/// A recorded operation from the dependency-free mock engine.
#[cfg(feature = "mock")]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MockCall {
    Connect {
        remote_engine_id: String,
        timeout_ms: i32,
    },
    RegisterMem {
        addr: usize,
        size: usize,
    },
    DeregisterMem {
        addr: usize,
    },
    TransferRead {
        remote_engine_id: String,
        local_addr: usize,
        remote_addr: usize,
        len: usize,
        timeout_ms: i32,
    },
    TransferWrite {
        remote_engine_id: String,
        local_addr: usize,
        remote_addr: usize,
        len: usize,
        timeout_ms: i32,
    },
    Cleanup,
}

/// Dependency-free HiXL engine used by ordinary Rust CI.
#[cfg(feature = "mock")]
#[derive(Debug)]
pub struct HixlEngine {
    dev: i32,
    calls: std::sync::Arc<std::sync::Mutex<Vec<MockCall>>>,
}

#[cfg(feature = "mock")]
impl HixlEngine {
    pub fn new(dev: i32, _engine_id: &str) -> Result<Self, String> {
        Ok(Self {
            dev,
            calls: std::sync::Arc::new(std::sync::Mutex::new(Vec::new())),
        })
    }

    pub fn ptr(&self) -> *mut c_void {
        std::sync::Arc::as_ptr(&self.calls).cast_mut().cast()
    }

    pub fn dev(&self) -> i32 {
        self.dev
    }

    pub fn acl_ctx(&self) -> usize {
        0
    }

    pub fn connect(&self, remote_engine_id: &str, timeout_ms: i32) -> Result<(), i32> {
        self.record(MockCall::Connect {
            remote_engine_id: remote_engine_id.to_string(),
            timeout_ms,
        });
        Ok(())
    }

    pub fn register_mem(&self, addr: usize, size: usize) -> Result<(), i32> {
        self.record(MockCall::RegisterMem { addr, size });
        Ok(())
    }

    pub fn deregister_mem(&self, addr: usize) -> Result<(), i32> {
        self.record(MockCall::DeregisterMem { addr });
        Ok(())
    }

    pub fn transfer_read(
        &self,
        remote_engine_id: &str,
        local_addr: usize,
        remote_addr: usize,
        len: usize,
        timeout_ms: i32,
    ) -> Result<(), i32> {
        self.record(MockCall::TransferRead {
            remote_engine_id: remote_engine_id.to_string(),
            local_addr,
            remote_addr,
            len,
            timeout_ms,
        });
        Ok(())
    }

    pub fn transfer_write(
        &self,
        remote_engine_id: &str,
        local_addr: usize,
        remote_addr: usize,
        len: usize,
        timeout_ms: i32,
    ) -> Result<(), i32> {
        self.record(MockCall::TransferWrite {
            remote_engine_id: remote_engine_id.to_string(),
            local_addr,
            remote_addr,
            len,
            timeout_ms,
        });
        Ok(())
    }

    pub fn calls(&self) -> Vec<MockCall> {
        self.calls.lock().unwrap().clone()
    }

    pub fn call_log(&self) -> std::sync::Arc<std::sync::Mutex<Vec<MockCall>>> {
        self.calls.clone()
    }

    fn record(&self, call: MockCall) {
        self.calls.lock().unwrap().push(call);
    }
}

#[cfg(feature = "mock")]
impl Drop for HixlEngine {
    fn drop(&mut self) {
        self.record(MockCall::Cleanup);
    }
}

#[cfg(feature = "mock")]
pub fn probe_hccs(_dev: i32) -> bool {
    true
}

#[cfg(all(test, feature = "mock"))]
mod tests {
    use super::*;

    #[test]
    fn mock_records_operations_and_cleanup() {
        let engine = HixlEngine::new(3, "local").unwrap();
        let log = engine.call_log();
        engine.register_mem(0x1000, 4096).unwrap();
        engine.connect("remote", 10_000).unwrap();
        engine
            .transfer_read("remote", 0x1000, 0x2000, 4096, 500)
            .unwrap();
        engine.deregister_mem(0x1000).unwrap();
        drop(engine);

        assert_eq!(
            *log.lock().unwrap(),
            vec![
                MockCall::RegisterMem {
                    addr: 0x1000,
                    size: 4096,
                },
                MockCall::Connect {
                    remote_engine_id: "remote".to_string(),
                    timeout_ms: 10_000,
                },
                MockCall::TransferRead {
                    remote_engine_id: "remote".to_string(),
                    local_addr: 0x1000,
                    remote_addr: 0x2000,
                    len: 4096,
                    timeout_ms: 500,
                },
                MockCall::DeregisterMem { addr: 0x1000 },
                MockCall::Cleanup,
            ]
        );
    }
}
