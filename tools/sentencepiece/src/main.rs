fn main() -> Result<(), Box<dyn std::error::Error>> {
    let schema_dir = std::env::args_os().nth(1).ok_or("expected schema directory")?;
    let schema_dir = std::path::PathBuf::from(schema_dir);
    let output = "../../crates/model-package/src/generated";
    prost_build::Config::new()
        .disable_comments(["."])
        .out_dir(output)
        .compile_protos(&[schema_dir.join("sentencepiece_model.proto")], &[schema_dir])?;
    let status = std::process::Command::new("rustfmt")
        .args(["--edition", "2024", &format!("{output}/sentencepiece.rs")])
        .status()?;
    if !status.success() {
        return Err("rustfmt failed".into());
    }
    Ok(())
}
