/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Build script for hixl-sys.
//!
//! Compiles `cpp/hixl_shim.cpp` into a static library and links it
//! together with the CANN runtime libraries (libcann_hixl, libascendcl).

fn main() {
    if std::env::var_os("CARGO_FEATURE_MOCK").is_some() {
        return;
    }

    let config = build_utils::ascend::discover_ascend_config()
        .expect("Ascend CANN installation not found — see build_utils::ascend for details");

    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let shim_src = format!("{}/cpp/hixl_shim.cpp", manifest_dir);

    println!("cargo:rerun-if-changed=cpp/hixl_shim.cpp");

    cc::Build::new()
        .cpp(true)
        .file(&shim_src)
        .flag("-std=c++17")
        .flag("-fPIC")
        .flag("-O2")
        .include(&config.include_dir)
        .compile("hixl_shim");

    // Dynamic link to CANN runtime libraries
    println!(
        "cargo:rustc-link-search=native={}",
        config.lib_dir.display()
    );
    println!("cargo:rustc-link-lib=dylib=cann_hixl");
    println!("cargo:rustc-link-lib=dylib=ascendcl");

    // Ensure the CANN lib dir is in the runtime library search path
    println!(
        "cargo:rustc-link-arg=-Wl,-rpath,{}",
        config.lib_dir.display()
    );

    // Link libstdc++ (the shim is C++)
    build_utils::link_libstdcpp_static();
}
