#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenKind {
    Keyword,
    Ident,
    IntLiteral,
    FloatLiteral,
    StringLiteral,
    CharLiteral,
    Lifetime,
    Operator,
    Delimiter,
    Punctuation,
    LineComment,
    BlockComment,
    DocComment,
    Whitespace,
    Unknown,
}

impl TokenKind {
    pub fn color_code(self) -> &'static str {
        match self {
            TokenKind::Keyword => "\x1b[1;34m",     // bold blue
            TokenKind::Ident => "\x1b[0m",           // default
            TokenKind::IntLiteral => "\x1b[33m",     // yellow
            TokenKind::FloatLiteral => "\x1b[33m",   // yellow
            TokenKind::StringLiteral => "\x1b[32m",  // green
            TokenKind::CharLiteral => "\x1b[32m",    // green
            TokenKind::Lifetime => "\x1b[36m",       // cyan
            TokenKind::Operator => "\x1b[1;35m",     // bold magenta
            TokenKind::Delimiter => "\x1b[1;37m",    // bold white
            TokenKind::Punctuation => "\x1b[35m",    // magenta
            TokenKind::LineComment => "\x1b[2;37m",  // dim
            TokenKind::BlockComment => "\x1b[2;37m", // dim
            TokenKind::DocComment => "\x1b[2;32m",   // dim green
            TokenKind::Whitespace => "",
            TokenKind::Unknown => "\x1b[1;31m",      // bold red
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            TokenKind::Keyword => "Keyword",
            TokenKind::Ident => "Ident",
            TokenKind::IntLiteral => "Int",
            TokenKind::FloatLiteral => "Float",
            TokenKind::StringLiteral => "String",
            TokenKind::CharLiteral => "Char",
            TokenKind::Lifetime => "Lifetime",
            TokenKind::Operator => "Operator",
            TokenKind::Delimiter => "Delimiter",
            TokenKind::Punctuation => "Punct",
            TokenKind::LineComment => "Comment",
            TokenKind::BlockComment => "Comment",
            TokenKind::DocComment => "DocComment",
            TokenKind::Whitespace => "Whitespace",
            TokenKind::Unknown => "Unknown",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Span {
    pub start: usize,
    pub end: usize,
    pub line: usize,
    pub col: usize,
}

#[derive(Debug, Clone)]
pub struct Token {
    pub kind: TokenKind,
    pub text: String,
    pub span: Span,
}

const KEYWORDS: &[&str] = &[
    "as", "async", "await", "break", "const", "continue", "crate", "dyn", "else", "enum",
    "extern", "false", "fn", "for", "if", "impl", "in", "let", "loop", "match", "mod", "move",
    "mut", "pub", "ref", "return", "self", "Self", "static", "struct", "super", "trait", "true",
    "type", "union", "unsafe", "use", "where", "while", "yield", "abstract", "become", "box",
    "do", "final", "macro", "override", "priv", "try", "typeof", "unsized", "virtual",
];

pub struct Lexer {
    src: Vec<char>,
    pos: usize,
    line: usize,
    col: usize,
}

impl Lexer {
    pub fn new(source: &str) -> Self {
        Self {
            src: source.chars().collect(),
            pos: 0,
            line: 1,
            col: 1,
        }
    }

    pub fn tokenize(&mut self) -> Vec<Token> {
        let mut tokens = Vec::new();
        while self.pos < self.src.len() {
            tokens.push(self.next_token());
        }
        tokens
    }

    fn peek(&self) -> Option<char> {
        self.src.get(self.pos).copied()
    }

    fn peek_ahead(&self, offset: usize) -> Option<char> {
        self.src.get(self.pos + offset).copied()
    }

    fn advance(&mut self) -> char {
        let ch = self.src[self.pos];
        self.pos += 1;
        if ch == '\n' {
            self.line += 1;
            self.col = 1;
        } else {
            self.col += 1;
        }
        ch
    }

    fn make_token(&self, kind: TokenKind, start: usize, start_line: usize, start_col: usize) -> Token {
        let text: String = self.src[start..self.pos].iter().collect();
        Token {
            kind,
            text,
            span: Span {
                start,
                end: self.pos,
                line: start_line,
                col: start_col,
            },
        }
    }

    fn next_token(&mut self) -> Token {
        let start = self.pos;
        let start_line = self.line;
        let start_col = self.col;
        let ch = self.advance();

        match ch {
            // Whitespace
            c if c.is_ascii_whitespace() => {
                while self.peek().is_some_and(|c| c.is_ascii_whitespace()) {
                    self.advance();
                }
                self.make_token(TokenKind::Whitespace, start, start_line, start_col)
            }

            // Comments or slash operator
            '/' => {
                match self.peek() {
                    Some('/') => {
                        self.advance();
                        let kind = if self.peek() == Some('/') || self.peek() == Some('!') {
                            TokenKind::DocComment
                        } else {
                            TokenKind::LineComment
                        };
                        while self.peek().is_some_and(|c| c != '\n') {
                            self.advance();
                        }
                        self.make_token(kind, start, start_line, start_col)
                    }
                    Some('*') => {
                        self.advance();
                        let kind = if self.peek() == Some('*') || self.peek() == Some('!') {
                            TokenKind::DocComment
                        } else {
                            TokenKind::BlockComment
                        };
                        let mut depth = 1;
                        while depth > 0 && self.pos < self.src.len() {
                            if self.peek() == Some('/') && self.peek_ahead(1) == Some('*') {
                                self.advance();
                                self.advance();
                                depth += 1;
                            } else if self.peek() == Some('*') && self.peek_ahead(1) == Some('/') {
                                self.advance();
                                self.advance();
                                depth -= 1;
                            } else {
                                self.advance();
                            }
                        }
                        self.make_token(kind, start, start_line, start_col)
                    }
                    Some('=') => {
                        self.advance();
                        self.make_token(TokenKind::Operator, start, start_line, start_col)
                    }
                    _ => self.make_token(TokenKind::Operator, start, start_line, start_col),
                }
            }

            // String literals
            '"' => {
                while self.pos < self.src.len() {
                    match self.peek() {
                        Some('\\') => {
                            self.advance();
                            if self.pos < self.src.len() {
                                self.advance();
                            }
                        }
                        Some('"') => {
                            self.advance();
                            break;
                        }
                        _ => {
                            self.advance();
                        }
                    }
                }
                self.make_token(TokenKind::StringLiteral, start, start_line, start_col)
            }

            // Char literals
            '\'' if self.peek_ahead(1) == Some('\'')
                || (self.peek() == Some('\\') && self.peek_ahead(2) == Some('\'')) =>
            {
                if self.peek() == Some('\\') {
                    self.advance(); // backslash
                    self.advance(); // escaped char
                } else {
                    self.advance(); // the char
                }
                if self.peek() == Some('\'') {
                    self.advance();
                }
                self.make_token(TokenKind::CharLiteral, start, start_line, start_col)
            }

            // Lifetime or char literal
            '\'' => {
                if self.peek().is_some_and(|c| c.is_ascii_alphabetic() || c == '_') {
                    while self.peek().is_some_and(|c| c.is_ascii_alphanumeric() || c == '_') {
                        self.advance();
                    }
                    self.make_token(TokenKind::Lifetime, start, start_line, start_col)
                } else {
                    self.make_token(TokenKind::Punctuation, start, start_line, start_col)
                }
            }

            // Raw string: r"..." or r#"..."#
            'r' if self.peek() == Some('"') || self.peek() == Some('#') => {
                let mut hashes = 0;
                while self.peek() == Some('#') {
                    self.advance();
                    hashes += 1;
                }
                if self.peek() == Some('"') {
                    self.advance();
                    'outer: loop {
                        if self.pos >= self.src.len() {
                            break;
                        }
                        if self.peek() == Some('"') {
                            self.advance();
                            let mut closing_hashes = 0;
                            while closing_hashes < hashes && self.peek() == Some('#') {
                                self.advance();
                                closing_hashes += 1;
                            }
                            if closing_hashes == hashes {
                                break 'outer;
                            }
                        } else {
                            self.advance();
                        }
                    }
                    self.make_token(TokenKind::StringLiteral, start, start_line, start_col)
                } else {
                    // Not a raw string, treat as identifier
                    self.lex_ident_rest(start, start_line, start_col)
                }
            }

            // Byte string: b"..." or byte char: b'...'
            'b' if self.peek() == Some('"') || self.peek() == Some('\'') => {
                let quote = self.advance();
                while self.pos < self.src.len() {
                    match self.peek() {
                        Some('\\') => {
                            self.advance();
                            if self.pos < self.src.len() {
                                self.advance();
                            }
                        }
                        Some(c) if c == quote => {
                            self.advance();
                            break;
                        }
                        _ => {
                            self.advance();
                        }
                    }
                }
                let kind = if quote == '"' {
                    TokenKind::StringLiteral
                } else {
                    TokenKind::CharLiteral
                };
                self.make_token(kind, start, start_line, start_col)
            }

            // Numbers
            c if c.is_ascii_digit() => {
                let mut is_float = false;
                // Hex, octal, binary
                if c == '0' {
                    match self.peek() {
                        Some('x' | 'X') => {
                            self.advance();
                            while self
                                .peek()
                                .is_some_and(|c| c.is_ascii_hexdigit() || c == '_')
                            {
                                self.advance();
                            }
                            return self.make_token(
                                TokenKind::IntLiteral,
                                start,
                                start_line,
                                start_col,
                            );
                        }
                        Some('o' | 'O') => {
                            self.advance();
                            while self.peek().is_some_and(|c| ('0'..='7').contains(&c) || c == '_')
                            {
                                self.advance();
                            }
                            return self.make_token(
                                TokenKind::IntLiteral,
                                start,
                                start_line,
                                start_col,
                            );
                        }
                        Some('b' | 'B') => {
                            self.advance();
                            while self.peek().is_some_and(|c| c == '0' || c == '1' || c == '_') {
                                self.advance();
                            }
                            return self.make_token(
                                TokenKind::IntLiteral,
                                start,
                                start_line,
                                start_col,
                            );
                        }
                        _ => {}
                    }
                }
                // Decimal digits
                while self.peek().is_some_and(|c| c.is_ascii_digit() || c == '_') {
                    self.advance();
                }
                // Float: decimal point followed by digit
                if self.peek() == Some('.')
                    && self.peek_ahead(1).is_some_and(|c| c.is_ascii_digit())
                {
                    is_float = true;
                    self.advance(); // '.'
                    while self.peek().is_some_and(|c| c.is_ascii_digit() || c == '_') {
                        self.advance();
                    }
                }
                // Exponent
                if self.peek().is_some_and(|c| c == 'e' || c == 'E') {
                    is_float = true;
                    self.advance();
                    if self.peek().is_some_and(|c| c == '+' || c == '-') {
                        self.advance();
                    }
                    while self.peek().is_some_and(|c| c.is_ascii_digit() || c == '_') {
                        self.advance();
                    }
                }
                // Type suffix: u8, i32, f64, etc.
                if self.peek().is_some_and(|c| c == 'u' || c == 'i' || c == 'f') {
                    let suffix_start = self.pos;
                    self.advance();
                    while self.peek().is_some_and(|c| c.is_ascii_digit()) {
                        self.advance();
                    }
                    // Check if suffix contains 'f' -> float
                    if self.src[suffix_start] == 'f' {
                        is_float = true;
                    }
                }
                let kind = if is_float {
                    TokenKind::FloatLiteral
                } else {
                    TokenKind::IntLiteral
                };
                self.make_token(kind, start, start_line, start_col)
            }

            // Identifiers and keywords
            c if c.is_ascii_alphabetic() || c == '_' => {
                self.lex_ident_rest(start, start_line, start_col)
            }

            // Delimiters
            '(' | ')' | '{' | '}' | '[' | ']' => {
                self.make_token(TokenKind::Delimiter, start, start_line, start_col)
            }

            // Multi-char operators and punctuation
            '=' => {
                if self.peek() == Some('=') {
                    self.advance();
                } else if self.peek() == Some('>') {
                    self.advance();
                }
                self.make_token(TokenKind::Operator, start, start_line, start_col)
            }
            '!' => {
                if self.peek() == Some('=') {
                    self.advance();
                }
                self.make_token(TokenKind::Operator, start, start_line, start_col)
            }
            '<' => {
                if self.peek() == Some('=') {
                    self.advance();
                } else if self.peek() == Some('<') {
                    self.advance();
                    if self.peek() == Some('=') {
                        self.advance();
                    }
                }
                self.make_token(TokenKind::Operator, start, start_line, start_col)
            }
            '>' => {
                if self.peek() == Some('=') {
                    self.advance();
                } else if self.peek() == Some('>') {
                    self.advance();
                    if self.peek() == Some('=') {
                        self.advance();
                    }
                }
                self.make_token(TokenKind::Operator, start, start_line, start_col)
            }
            '+' | '*' | '%' | '^' => {
                if self.peek() == Some('=') {
                    self.advance();
                }
                self.make_token(TokenKind::Operator, start, start_line, start_col)
            }
            '-' => {
                if self.peek() == Some('=') || self.peek() == Some('>') {
                    self.advance();
                }
                self.make_token(TokenKind::Operator, start, start_line, start_col)
            }
            '&' => {
                if self.peek() == Some('&') || self.peek() == Some('=') {
                    self.advance();
                }
                self.make_token(TokenKind::Operator, start, start_line, start_col)
            }
            '|' => {
                if self.peek() == Some('|') || self.peek() == Some('=') {
                    self.advance();
                }
                self.make_token(TokenKind::Operator, start, start_line, start_col)
            }
            '.' => {
                if self.peek() == Some('.') {
                    self.advance();
                    if self.peek() == Some('=') || self.peek() == Some('.') {
                        self.advance();
                    }
                }
                self.make_token(TokenKind::Operator, start, start_line, start_col)
            }
            ':' => {
                if self.peek() == Some(':') {
                    self.advance();
                }
                self.make_token(TokenKind::Punctuation, start, start_line, start_col)
            }

            // Simple punctuation
            ';' | ',' | '#' | '$' | '@' | '?' | '~' => {
                self.make_token(TokenKind::Punctuation, start, start_line, start_col)
            }

            _ => self.make_token(TokenKind::Unknown, start, start_line, start_col),
        }
    }

    fn lex_ident_rest(&mut self, start: usize, start_line: usize, start_col: usize) -> Token {
        while self.peek().is_some_and(|c| c.is_ascii_alphanumeric() || c == '_') {
            self.advance();
        }
        // Check if it's a macro invocation identifier (the `!` is handled separately)
        let text: String = self.src[start..self.pos].iter().collect();
        let kind = if KEYWORDS.contains(&text.as_str()) {
            TokenKind::Keyword
        } else {
            TokenKind::Ident
        };
        Token {
            kind,
            text,
            span: Span {
                start,
                end: self.pos,
                line: start_line,
                col: start_col,
            },
        }
    }
}
