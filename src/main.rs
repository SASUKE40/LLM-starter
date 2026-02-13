mod lexer;

use lexer::{Lexer, TokenKind};
use std::{env, fs};

const RESET: &str = "\x1b[0m";

const SAMPLE: &str = r#"use std::collections::HashMap;

/// A simple key-value store.
pub struct Store<V> {
    data: HashMap<String, V>,
}

impl<V: Clone> Store<V> {
    pub fn new() -> Self {
        Self {
            data: HashMap::new(),
        }
    }

    /// Insert a value, returning the old one if present.
    pub fn insert(&mut self, key: &str, value: V) -> Option<V> {
        self.data.insert(key.to_string(), value)
    }

    pub fn get(&self, key: &str) -> Option<&V> {
        self.data.get(key)
    }
}

fn main() {
    let mut store = Store::new();
    store.insert("hello", 42);
    let val = store.get("hello");
    // Print the result
    if let Some(v) = val {
        println!("value = {v}");
    }
    let bits = 0b1010_1100;
    let hex = 0xFF;
    let pi = 3.14159f64;
    let ch = 'A';
    let escaped = "line1\nline2";
    let _unused = bits + hex;
    assert!(pi > 3.0, "pi should be > 3");
    println!("{ch} {escaped} {_unused}");
}
"#;

fn print_highlighted(source: &str) {
    let mut lex = Lexer::new(source);
    let tokens = lex.tokenize();

    println!("\x1b[1;4m Syntax Highlighted Output \x1b[0m\n");
    for tok in &tokens {
        let color = tok.kind.color_code();
        if tok.kind == TokenKind::Whitespace {
            print!("{}", tok.text);
        } else {
            print!("{}{}{}", color, tok.text, RESET);
        }
    }
    println!();
}

fn print_token_table(source: &str) {
    let mut lex = Lexer::new(source);
    let tokens = lex.tokenize();

    println!("\x1b[1;4m Token Table \x1b[0m\n");
    println!(
        "{:<12} {:<6} {:<5} {:<6} {}",
        "KIND", "LINE", "COL", "LEN", "TEXT"
    );
    println!("{}", "-".repeat(72));

    for tok in &tokens {
        if tok.kind == TokenKind::Whitespace {
            continue;
        }
        let display_text = tok
            .text
            .replace('\n', "\\n")
            .replace('\r', "\\r")
            .replace('\t', "\\t");
        let truncated = if display_text.len() > 40 {
            format!("{}...", &display_text[..37])
        } else {
            display_text
        };
        println!(
            "{}{:<12}{} {:<6} {:<5} {:<6} {}",
            tok.kind.color_code(),
            tok.kind.label(),
            RESET,
            tok.span.line,
            tok.span.col,
            tok.text.len(),
            truncated,
        );
    }
}

fn print_stats(source: &str) {
    let mut lex = Lexer::new(source);
    let tokens = lex.tokenize();

    let mut counts = std::collections::HashMap::new();
    for tok in &tokens {
        if tok.kind == TokenKind::Whitespace {
            continue;
        }
        *counts.entry(tok.kind.label()).or_insert(0usize) += 1;
    }

    println!("\n\x1b[1;4m Token Statistics \x1b[0m\n");
    let mut entries: Vec<_> = counts.into_iter().collect();
    entries.sort_by(|a, b| b.1.cmp(&a.1));
    for (label, count) in &entries {
        let bar = "#".repeat((*count).min(40));
        println!("  {:<12} {:>4}  \x1b[36m{}\x1b[0m", label, count, bar);
    }
    let total: usize = entries.iter().map(|(_, c)| c).sum();
    println!("\n  {:<12} {:>4}", "TOTAL", total);
}

fn main() {
    let args: Vec<String> = env::args().collect();

    let source = if args.len() > 1 {
        fs::read_to_string(&args[1]).unwrap_or_else(|e| {
            eprintln!("Error reading {}: {e}", args[1]);
            std::process::exit(1);
        })
    } else {
        println!("\x1b[2m(No file provided, using built-in sample Rust code)\x1b[0m\n");
        SAMPLE.to_string()
    };

    print_highlighted(&source);
    print_token_table(&source);
    print_stats(&source);
}
