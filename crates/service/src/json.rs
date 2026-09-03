// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 Yuriy Krasilnikov
// Copyright (C) 2026 Deeray Ltd.
//
// This file is part of GigaAM v3 Runtime, free software distributed under
// the terms of the GNU Lesser General Public License, either version 3 of the
// License, or (at your option) any later version. See COPYING.LESSER and COPYING
// for the full terms. There is NO WARRANTY, to the extent permitted by law.

//! Minimal dependency-free JSON writer. The service emits a fixed response schema and
//! accepts no JSON input (request bodies contain audio and parameters are in the query),
//! so it does not need a parser. UTF-8 is emitted as-is; only `"`, `\`, and control
//! characters are escaped.
use gigaam_primitives::integral_f64_to_i64;

/// Shared media type for every Service JSON response envelope.
pub(crate) const JSON_CONTENT_TYPE: &str = "application/json; charset=utf-8";

#[derive(Debug, Clone, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    /// Integer (for counters and channels), without a decimal point.
    Int(i64),
    /// Unsigned integer, preserved exactly rather than coerced to a signed range.
    UInt(u64),
    /// Platform-sized unsigned integer, preserved exactly rather than coerced to a signed range.
    Usize(usize),
    /// Wide unsigned integer used for exact duration counters.
    UInt128(u128),
    /// Floating-point value; NaN and ±Inf become null because JSON has no representation
    /// for them.
    Num(f64),
    Str(String),
    Array(Vec<Json>),
    /// Object whose key order is preserved for deterministic output.
    Object(Vec<(String, Json)>),
}

impl Json {
    pub fn str(s: impl Into<String>) -> Json {
        Json::Str(s.into())
    }

    /// Floating-point value with a fixed maximum number of decimal places (timestamps use
    /// two).
    pub fn round(x: f32, digits: i32) -> Json {
        if !x.is_finite() {
            return Json::Null;
        }
        let f = 10f64.powi(digits);
        Json::Num((f64::from(x) * f).round() / f)
    }

    fn write(&self, out: &mut String) {
        match self {
            Json::Null => out.push_str("null"),
            Json::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
            Json::Int(i) => out.push_str(&i.to_string()),
            Json::UInt(i) => out.push_str(&i.to_string()),
            Json::Usize(i) => out.push_str(&i.to_string()),
            Json::UInt128(i) => out.push_str(&i.to_string()),
            Json::Num(n) => {
                if !n.is_finite() {
                    out.push_str("null");
                } else if *n == n.trunc() && n.abs() < 1e15 {
                    // Omit ".0" from integer-valued numbers so 2.0 becomes "2".
                    match integral_f64_to_i64(*n) {
                        Ok(integer) => out.push_str(&integer.to_string()),
                        Err(_) => out.push_str(&format!("{n}")),
                    }
                } else {
                    out.push_str(&format!("{n}"));
                }
            }
            Json::Str(s) => write_str(s, out),
            Json::Array(a) => {
                out.push('[');
                for (i, v) in a.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    v.write(out);
                }
                out.push(']');
            }
            Json::Object(o) => {
                out.push('{');
                for (i, (k, v)) in o.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    write_str(k, out);
                    out.push(':');
                    v.write(out);
                }
                out.push('}');
            }
        }
    }
}

impl std::fmt::Display for Json {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut out = String::new();
        self.write(&mut out);
        f.write_str(&out)
    }
}

fn write_str(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0C}' => out.push_str("\\f"),
            c if u32::from(c) < 0x20 => out.push_str(&format!("\\u{:04x}", u32::from(c))),
            c => out.push(c), // Preserve UTF-8, including Cyrillic, as-is.
        }
    }
    out.push('"');
}

/// Convenience object constructor: `obj([("a", Json::Int(1))])`.
pub fn obj(fields: Vec<(&str, Json)>) -> Json {
    Json::Object(
        fields
            .into_iter()
            .map(|(k, v)| (k.to_string(), v))
            .collect(),
    )
}

/// The one canonical Service error object, preserving the public field order.
pub(crate) fn error_object(message: impl Into<String>) -> String {
    obj(vec![("error", Json::str(message))]).to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn escapes_and_unicode() {
        assert_eq!(Json::str("a\"b\\c\nd\t").to_string(), r#""a\"b\\c\nd\t""#);
        assert_eq!(Json::str("Hello, café").to_string(), "\"Hello, café\""); // UTF-8 is preserved.
        let mut s = String::new();
        write_str("\u{01}", &mut s);
        assert_eq!(s, "\"\\u0001\""); // control char -> \u0001
    }
    #[test]
    fn numbers_int_vs_float() {
        assert_eq!(Json::Num(2.0).to_string(), "2");
        assert_eq!(Json::Int(2).to_string(), "2");
        assert_eq!(Json::UInt(u64::MAX).to_string(), u64::MAX.to_string());
        assert_eq!(Json::Usize(usize::MAX).to_string(), usize::MAX.to_string());
        assert_eq!(Json::UInt128(u128::MAX).to_string(), u128::MAX.to_string());
        assert_eq!(Json::round(0.04321, 2).to_string(), "0.04");
        assert_eq!(Json::Num(f64::NAN).to_string(), "null");
        assert_eq!(Json::round(f32::INFINITY, 2).to_string(), "null");
    }
    #[test]
    fn object_and_array_order() {
        let j = obj(vec![
            ("text", Json::str("one two")),
            (
                "words",
                Json::Array(vec![obj(vec![
                    ("start", Json::round(0.04, 2)),
                    ("text", Json::str("one")),
                ])]),
            ),
            ("n", Json::Int(2)),
            ("ok", Json::Bool(true)),
        ]);
        assert_eq!(
            j.to_string(),
            r#"{"text":"one two","words":[{"start":0.04,"text":"one"}],"n":2,"ok":true}"#
        );
    }
}
