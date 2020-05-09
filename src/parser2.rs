use crate::lexer2::Lexer;
use std::iter;
use std::error;
use std::fmt;
use std::io::{BufReader, Read};
use std::fs::File;
use std::path::Path;


#[derive(Clone, Debug)]
pub enum ObjError {
    Source,
    SourceDoesNotExist(String),
    Parse(ParseError),
}

impl fmt::Display for ObjError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            ObjError::Source => {
                write!(f, "An error occurred in the OBJ source.")
            }
            ObjError::SourceDoesNotExist(source) => {
                write!(f, "Source could not be found: {}.", source)
            }
            ObjError::Parse(parse_error) => {
                write!(f, "{}", parse_error)
            }
        }
    }
}

impl error::Error for ObjError {}


/// Parse a wavefront object file from a file buffer or other `Read` instance.
pub fn parse<F: Read>(file: F) -> Result<Object, ObjError> {
    let mut reader = BufReader::new(file);
    let mut string = String::new();
    reader.read_to_string(&mut string).unwrap();

    let mut parser = Parser::new(string.chars());

    match parser.parse() {
        Ok(obj_set) => Ok(obj_set),
        Err(e) => Err(ObjError::Parse(e)),
    }
}


/// Parse a wavefront object file from a file path.
pub fn parse_file<P: AsRef<Path>>(path: P) -> Result<Object, ObjError> {
    if !path.as_ref().exists() {
        let disp = path.as_ref().display();

        return Err(ObjError::SourceDoesNotExist(format!("{}", disp)));
    }

    let file = match File::open(path) {
        Ok(val) => val,
        Err(_) => return Err(ObjError::Source)
    };

    parse(file)
}


/// Parse a wavefront object file from a string.
pub fn parse_str(st: &str) -> Result<Object, ParseError> {
    let mut parser = Parser::new(st.chars());
    parser.parse()
}


#[inline]
fn slice(st: &Option<String>) -> Option<&str> {
    st.as_ref().map(|st| &st[..])
}

#[inline]
fn slice_res(st: &Result<String, ParseError>) -> Result<&str, &ParseError> {
    st.as_ref().map(|s| &s[..])
}


#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ErrorKind {
    EndOfFile,
    ExpectedStatementButGot(String, String),
    ExpectedFloatButGot(String),
    ExpectedIntegerButGot(String),
    ExpectedVertexIndexButGot(String),
    ExpectedTextureIndexButGot(String),
    ExpectedNormalIndexButGot(String),
    ExpectedVertexNormalIndexButGot(String),
    ExpectedVertexTextureIndexButGot(String),
    ExpectedVertexTextureNormalIndexButGot(String),
    EveryFaceElementMustHaveAtLeastThreeVertices,
    EveryVTNIndexMustHaveTheSameFormForAGivenFace,
    InvalidElementDeclaration(String),
    InvalidElement,
}

impl fmt::Display for ErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "")
    }
}

/// An error that is returned from parsing an invalid *.obj file, or
/// another kind of error.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParseError {
    line_number: usize,
    kind: ErrorKind,
}

impl ParseError {
    /// Generate a new parse error.
    fn new(line_number: usize, kind: ErrorKind) -> ParseError {
        ParseError {
            line_number: line_number,
            kind: kind,
        }
    }
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(f, "Parse error at line {}: {}", self.line_number, self.kind)
    }
}

impl error::Error for ParseError {}


#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Vertex {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

impl Vertex {
    pub fn new(x: f32, y: f32, z: f32, w: f32) -> Vertex {
        Vertex { x: x, y: y, z: z, w: w }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TextureVertex {
    pub u: f32,
    pub v: f32,
    pub w: f32,
}

impl TextureVertex {
    pub fn new(u: f32, v: f32, w: f32) -> TextureVertex {
        TextureVertex { u: u, v: v, w: w }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NormalVertex {
    pub i: f32,
    pub j: f32,
    pub k: f32,
}

impl NormalVertex {
    pub fn new(i: f32, j: f32, k: f32) -> NormalVertex {
        NormalVertex { i: i, j: j, k: k }
    }
}

type ElementIndex = u32;
type VertexIndex = u32;
type TextureVertexIndex = u32;
type NormalVertexIndex = u32;

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum VTNIndex { 
    V(VertexIndex),
    VT(VertexIndex, TextureVertexIndex), 
    VN(VertexIndex, NormalVertexIndex),
    VTN(VertexIndex, TextureVertexIndex, NormalVertexIndex),
}

impl VTNIndex {
    pub fn has_same_type_as(&self, other: &VTNIndex) -> bool {
        match (self, other) {
            (&VTNIndex::V(_),   &VTNIndex::V(_)) |
            (&VTNIndex::VT(_,_),  &VTNIndex::VT(_,_)) | 
            (&VTNIndex::VN(_,_),  &VTNIndex::VN(_,_)) | 
            (&VTNIndex::VTN(_,_,_), &VTNIndex::VTN(_,_,_)) => true,
            _ => false,
        }
    }
}

#[derive(Clone, Debug)]
pub enum VTNTriple<'a> {
    V(&'a Vertex),
    VT(&'a Vertex, &'a TextureVertex), 
    VN(&'a Vertex, &'a NormalVertex),
    VTN(&'a Vertex, &'a TextureVertex, &'a NormalVertex),
}

#[derive(Clone, Debug, PartialEq)]
pub enum Element {
    Point(VTNIndex),
    Face(VTNIndex, VTNIndex, VTNIndex),
}


#[derive(Clone, Debug, PartialEq)]
pub struct Object {
    pub vertex_set: Vec<Vertex>,
    pub texture_vertex_set: Vec<TextureVertex>,
    pub normal_vertex_set: Vec<NormalVertex>,
    pub elements: Vec<Element>,
}

/// A Wavefront OBJ file parser.
pub struct Parser<Stream> where Stream: Iterator<Item=char> {
    line_number: usize,
    lexer: iter::Peekable<Lexer<Stream>>,
    vtn_indices: Vec<VTNIndex>,
}

impl<Stream> Parser<Stream> where Stream: Iterator<Item=char> {
    pub fn new(input: Stream) -> Parser<Stream> {
        Parser {
            line_number: 1,
            lexer: Lexer::new(input).peekable(),
            vtn_indices: vec![],
        }
    }

    fn peek(&mut self) -> Option<&str> {
        self.lexer.peek().map(|token| token.content.as_str())
    }

    fn next(&mut self) -> Option<String> {
        let token = self.lexer.next().map(|t| t.content);
        if let Some(ref val) = token {
            if val == "\n" {
                self.line_number += 1;
            }
        }

        token
    }

    fn advance(&mut self) {
        self.next();
    }

    #[inline]
    fn error<T>(&self, kind: ErrorKind) -> Result<T, ParseError> {
        Err(ParseError::new(self.line_number, kind))
    }

    fn next_string(&mut self) -> Result<String, ParseError> {
        match self.next() {
            Some(st) => Ok(st),
            None => self.error(ErrorKind::EndOfFile)
        }
    }

    fn expect(&mut self, tag: &str) -> Result<String, ParseError> {
        let st = self.next_string()?;
        match st == tag {
            true => Ok(st),
            false => self.error(ErrorKind::ExpectedStatementButGot(tag.into(), st))
        }
    }

    fn parse_f32(&mut self) -> Result<f32, ParseError> {
        let st = self.next_string()?;
        match st.parse::<f32>() {
            Ok(val) => Ok(val),
            Err(_) => self.error(ErrorKind::ExpectedFloatButGot(st)),
        }
    }

    fn parse_u32(&mut self) -> Result<u32, ParseError> {
        let st = self.next_string()?;
        match st.parse::<u32>() {
            Ok(val) => Ok(val),
            Err(_) => self.error(ErrorKind::ExpectedIntegerButGot(st)),
        }
    }

    fn try_once<P, T>(&mut self, parser: P) -> Option<T> where P: FnOnce(&str) -> Option<T> {
        match self.peek() {
            Some(st) => parser(&st).map(|got| { self.advance(); got }),
            None => None,
        }
    }

    fn parse_vertex(&mut self) -> Result<Vertex, ParseError> {
        self.expect("v")?;
 
        let x = self.parse_f32()?;
        let y = self.parse_f32()?;
        let z = self.parse_f32()?;
        let mw = self.try_once(|st| st.parse::<f32>().ok());
        let w = mw.unwrap_or(1.0);

        Ok(Vertex { x: x, y: y, z: z, w: w })
    }

    fn parse_texture_vertex(&mut self) -> Result<TextureVertex, ParseError> {
        self.expect("vt")?;

        let u = self.parse_f32()?;
        let mv = self.try_once(|st| st.parse::<f32>().ok());
        let v = mv.unwrap_or(0.0);
        let mw = self.try_once(|st| st.parse::<f32>().ok());
        let w = mw.unwrap_or(0.0);

        Ok(TextureVertex { u: u, v: v, w: w })
    }

    fn parse_normal_vertex(&mut self) -> Result<NormalVertex, ParseError> {
        self.expect("vn")?;

        let i = self.parse_f32()?;
        let j = self.parse_f32()?;
        let k = self.parse_f32()?;

        Ok(NormalVertex { i: i, j: j, k: k })
    }

    fn skip_zero_or_more_newlines(&mut self) {
        loop {
            match self.peek() {
                Some("\n") => self.advance(),
                _ => break
            }
        }
    }

    fn skip_one_or_more_newlines(&mut self) -> Result<(), ParseError> {
        self.expect("\n")?;
        self.skip_zero_or_more_newlines();
        Ok(())
    }

    fn parse_vtn_index(&mut self) -> Result<VTNIndex, ParseError> {
        enum VTNErrorKind {
            ExpectedVertexIndexButGot,
            ExpectedNormalIndexButGot,
            ExpectedVertexNormalIndexButGot,
            ExpectedTextureIndexButGot,
            ExpectedVertexTextureIndexButGot,
        }

        fn parse_vn(st: &str) -> Result<VTNIndex, VTNErrorKind> {
            if let Some(v_index_in_str) = st.find("//") {
                let v_index = match st[0..v_index_in_str].parse::<u32>() {
                    Ok(val) => val,
                    Err(_) => return Err(VTNErrorKind::ExpectedVertexIndexButGot),
                };
                let vn_index = match st[v_index_in_str+2..].parse::<u32>() {
                    Ok(val) => val,
                    Err(_) => return Err(VTNErrorKind::ExpectedNormalIndexButGot),
                };
    
                return Ok(VTNIndex::VN(v_index, vn_index));
            } else {
                return Err(VTNErrorKind::ExpectedVertexNormalIndexButGot);
            }
        }
    
        fn parse_vt(st: &str) -> Result<VTNIndex, VTNErrorKind> {
            if let Some(v_index_in_str) = st.find("/") {
                let v_index = match st[0..v_index_in_str].parse::<u32>() {
                    Ok(val) => val,
                    Err(_) => return Err(VTNErrorKind::ExpectedVertexIndexButGot),
                };
                let vt_index = match st[v_index_in_str+1..].parse::<u32>() {
                    Ok(val) => val,
                    Err(_) => return Err(VTNErrorKind::ExpectedTextureIndexButGot)
                };
    
                return Ok(VTNIndex::VT(v_index, vt_index));
            } else {
                return Err(VTNErrorKind::ExpectedVertexTextureIndexButGot);
            }
        }
    
        fn parse_vtn(st: &str) -> Result<VTNIndex, VTNErrorKind> {
            let v_index_in_str = match st.find("/") {
                Some(val) => val,
                None => return Err(VTNErrorKind::ExpectedVertexIndexButGot),
            };
            let v_index = match st[0..v_index_in_str].parse::<u32>() {
                Ok(val) => val,
                Err(_) => return Err(VTNErrorKind::ExpectedVertexIndexButGot),
            };
            let vt_index_in_str = match st[(v_index_in_str + 1)..].find("/") {
                Some(val) => v_index_in_str + 1 + val,
                None => return Err(VTNErrorKind::ExpectedTextureIndexButGot),
            };
            let vt_index = match st[(v_index_in_str + 1)..vt_index_in_str].parse::<u32>() {
                Ok(val) => val,
                Err(_) => return Err(VTNErrorKind::ExpectedTextureIndexButGot),
            };
            let vn_index = match st[(vt_index_in_str + 1)..].parse::<u32>() {
                Ok(val) => val,
                Err(_) => return Err(VTNErrorKind::ExpectedNormalIndexButGot),
            };
       
            Ok(VTNIndex::VTN(v_index, vt_index, vn_index))
        }
    
        fn parse_v(st: &str) -> Result<VTNIndex, VTNErrorKind> {
            match st.parse::<u32>() {
                Ok(val) => Ok(VTNIndex::V(val)),
                Err(_) => return Err(VTNErrorKind::ExpectedVertexIndexButGot)
            }
        }


        let st = self.next_string()?;
        match parse_vn(&st) {
            Ok(val) => return Ok(val),
            Err(_) => {},
        }
        match parse_vtn(&st) {
            Ok(val) => return Ok(val),
            Err(_) => {},
        }
        match parse_vt(&st) {
            Ok(val) => return Ok(val),
            Err(_) => {},
        }
        match parse_v(&st) {
            Ok(val) => return Ok(val),
            Err(_) => {},
        }

        self.error(ErrorKind::ExpectedVertexTextureNormalIndexButGot(st.into()))
    }

    fn parse_vtn_indices(&mut self) -> Result<u32, ParseError> {
        let mut indices_parsed = 0;
        while let Ok(vtn_index) = self.parse_vtn_index() {
            self.vtn_indices.push(vtn_index);
            indices_parsed += 1;
        }

        Ok(indices_parsed)
    }

    fn parse_face(&mut self, elements: &mut Vec<Element>) -> Result<u32, ParseError> {
        self.expect("f")?;
        
        //let mut vtn_indices = vec![];
        self.vtn_indices.clear();
        //self.parse_vtn_indices(&mut vtn_indices)?;
        self.parse_vtn_indices()?;

        // Check that there are enough vtn indices.
        if self.vtn_indices.len() < 3 {
            return self.error(ErrorKind::EveryFaceElementMustHaveAtLeastThreeVertices);  
        }

        // Verify that each VTN index has the same type and if of a valid form.
        for i in 1..self.vtn_indices.len() {
            if !self.vtn_indices[i].has_same_type_as(&self.vtn_indices[0]) {
                return self.error(ErrorKind::EveryVTNIndexMustHaveTheSameFormForAGivenFace);
            }
        }

        // Triangulate the polygon with a triangle fan. Note that the OBJ specification
        // assumes that polygons are coplanar, and consequently the parser does not check
        // this. It is up to the model creator to ensure this.
        let vertex0 = self.vtn_indices[0];
        for i in 0..self.vtn_indices.len()-2 {
            elements.push(Element::Face(vertex0, self.vtn_indices[i+1], self.vtn_indices[i+2]));
        }

        Ok((self.vtn_indices.len() - 2) as u32)
    }

    fn parse_elements(&mut self, elements: &mut Vec<Element>) -> Result<u32, ParseError> {  
        match self.peek() {
            Some("f") => self.parse_face(elements),
            _ => self.error(ErrorKind::InvalidElement),
        }
    }

    fn parse_object(&mut self) -> Result<Object, ParseError> {
        let mut min_vertex_index = 1;
        let mut max_vertex_index = 1;
        let mut min_texture_index = 1;
        let mut max_texture_index = 1;
        let mut min_normal_index = 1;
        let mut max_normal_index = 1;

        let mut vertices: Vec<Vertex> = vec![];
        let mut texture_vertices = vec![];
        let mut normal_vertices = vec![];        
        let mut elements = vec![];
        loop {
            match self.peek() {
                Some("v")  => {
                    let vertex = self.parse_vertex()?;
                    vertices.push(vertex);
                }
                Some("vt") => {
                    let texture_vertex = self.parse_texture_vertex()?;
                    texture_vertices.push(texture_vertex);
                }
                Some("vn") => {
                    let normal_vertex = self.parse_normal_vertex()?;
                    normal_vertices.push(normal_vertex);
                }
                Some("f") => {
                    self.parse_elements(&mut elements)?;
                }
                Some("\n") => {
                    self.skip_one_or_more_newlines()?;
                }
                Some(other_st) => {
                    let st: String = other_st.into();
                    return self.error(ErrorKind::InvalidElementDeclaration(st));
                }
                None => {
                    break;
                }
            }
        }

        min_vertex_index  += vertices.len();
        max_vertex_index  += vertices.len();
        min_texture_index += texture_vertices.len();
        max_texture_index += texture_vertices.len();
        min_normal_index  += normal_vertices.len();
        max_normal_index  += normal_vertices.len();

        Ok(Object {
            vertex_set: vertices,
            texture_vertex_set: texture_vertices,
            normal_vertex_set: normal_vertices,
            elements: elements,
        })
    }

    pub fn parse(&mut self) -> Result<Object, ParseError> {
        self.parse_object()
    }
}


#[cfg(test)]
mod tests {
    use super::{
        Object,
        Vertex, 
        NormalVertex, 
        Element, 
        VTNIndex, 
    };

    fn test_case() -> (Result<Object, super::ParseError>, Result<Object, super::ParseError>) {
        let obj_file =r"                \
            v  0.0  0.0  0.0                  \
            v  0.0  0.0  1.0                  \
            v  0.0  1.0  0.0                  \
            v  0.0  1.0  1.0                  \
            v  1.0  0.0  0.0                  \
            v  1.0  0.0  1.0                  \
            v  1.0  1.0  0.0                  \
            v  1.0  1.0  1.0                  \
                                              \
            vn  0.0  0.0  1.0                 \
            vn  0.0  0.0 -1.0                 \
            vn  0.0  1.0  0.0                 \
            vn  0.0 -1.0  0.0                 \
            vn  1.0  0.0  0.0                 \
            vn -1.0  0.0  0.0                 \
                                              \
            f  1//2  7//2  5//2               \
            f  1//2  3//2  7//2               \
            f  1//6  4//6  3//6               \
            f  1//6  2//6  4//6               \
            f  3//3  8//3  7//3               \
            f  3//3  4//3  8//3               \
            f  5//5  7//5  8//5               \
            f  5//5  8//5  6//5               \
            f  1//4  5//4  6//4               \
            f  1//4  6//4  2//4               \
            f  2//1  6//1  8//1               \
            f  2//1  8//1  4//1               \
        ";
        let vertex_set = vec![
            Vertex { x: 0.0,  y: 0.0, z: 0.0, w: 1.0 },
            Vertex { x: 0.0,  y: 0.0, z: 1.0, w: 1.0 },
            Vertex { x: 0.0,  y: 1.0, z: 0.0, w: 1.0 },
            Vertex { x: 0.0,  y: 1.0, z: 1.0, w: 1.0 },
            Vertex { x: 1.0,  y: 0.0, z: 0.0, w: 1.0 },
            Vertex { x: 1.0,  y: 0.0, z: 1.0, w: 1.0 },
            Vertex { x: 1.0,  y: 1.0, z: 0.0, w: 1.0 },
            Vertex { x: 1.0,  y: 1.0, z: 1.0, w: 1.0 },
        ];
        let element_set = vec![
            Element::Face(VTNIndex::VN(1,2), VTNIndex::VN(7,2), VTNIndex::VN(5,2)),
            Element::Face(VTNIndex::VN(1,2), VTNIndex::VN(3,2), VTNIndex::VN(7,2)),
            Element::Face(VTNIndex::VN(1,6), VTNIndex::VN(4,6), VTNIndex::VN(3,6)),
            Element::Face(VTNIndex::VN(1,6), VTNIndex::VN(2,6), VTNIndex::VN(4,6)),
            Element::Face(VTNIndex::VN(3,3), VTNIndex::VN(8,3), VTNIndex::VN(7,3)),
            Element::Face(VTNIndex::VN(3,3), VTNIndex::VN(4,3), VTNIndex::VN(8,3)),
            Element::Face(VTNIndex::VN(5,5), VTNIndex::VN(7,5), VTNIndex::VN(8,5)),
            Element::Face(VTNIndex::VN(5,5), VTNIndex::VN(8,5), VTNIndex::VN(6,5)),
            Element::Face(VTNIndex::VN(1,4), VTNIndex::VN(5,4), VTNIndex::VN(6,4)),
            Element::Face(VTNIndex::VN(1,4), VTNIndex::VN(6,4), VTNIndex::VN(2,4)),
            Element::Face(VTNIndex::VN(2,1), VTNIndex::VN(6,1), VTNIndex::VN(8,1)),
            Element::Face(VTNIndex::VN(2,1), VTNIndex::VN(8,1), VTNIndex::VN(4,1)),
        ];
        let normal_vertex_set = vec![
            NormalVertex { i:  0.0, j:  0.0, k:  1.0 },
            NormalVertex { i:  0.0, j:  0.0, k: -1.0 },
            NormalVertex { i:  0.0, j:  1.0, k:  0.0 },
            NormalVertex { i:  0.0, j: -1.0, k:  0.0 },
            NormalVertex { i:  1.0, j:  0.0, k:  0.0 },
            NormalVertex { i: -1.0, j:  0.0, k:  0.0 },
        ];

        let expected = Object {
            vertex_set: vertex_set,
            texture_vertex_set:vec![],
            normal_vertex_set: normal_vertex_set,
            elements: element_set,
        };
        let mut parser = super::Parser::new(obj_file.chars());
        let result = parser.parse();

        (result, Ok(expected))
    }

    #[test]
    fn test_parse_object_set1() {
        let (result, expected) = test_case();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_parse_object_set1_tokenwise() {
        let (result, expected) = test_case();
        let result = result.unwrap();
        let expected = expected.unwrap();

        assert_eq!(result.vertex_set, expected.vertex_set);
        assert_eq!(result.texture_vertex_set, expected.texture_vertex_set);
        assert_eq!(result.normal_vertex_set, expected.normal_vertex_set);
        assert_eq!(result.elements, expected.elements);
    }
}