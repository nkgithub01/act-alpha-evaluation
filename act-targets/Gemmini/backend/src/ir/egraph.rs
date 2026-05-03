use egg::{Analysis, DidMerge, EGraph, FromOp, FromOpError, Id, Language, LanguageChildren};

use crate::ir::dtype::Dtype;

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Clone)]
pub enum TensorOp {
    // ISA instructions
    MvinSpad(String, [Id; 1]),
    MvoutSpad(String, [Id; 1]),
    MvoutSpadRelu(String, [Id; 1]),
    MvinAcc(String, [Id; 1]),
    MvoutAcc(String, [Id; 1]),
    MvoutAccRelu(String, [Id; 1]),
    MvinAccLow(String, [Id; 1]),
    MvinAccLowAdd(String, [Id; 2]),
    MvoutAccLow(String, [Id; 1]),
    MvoutAccLowRelu(String, [Id; 1]),
    Matmul8(String, [Id; 2]),
    Matmul32(String, [Id; 2]),
    Mac8(String, [Id; 3]),
    Mac32(String, [Id; 3]),
    // IR operators
    OpAdd([Id; 2]),
    OpBitcvt([Id; 1]),
    OpBroadcast(String, [Id; 1]),
    OpClamp([Id; 3]),
    OpConcat(String, [Id; 2]),
    OpConstant(String),
    OpConvert(String, [Id; 1]),
    OpCopy([Id; 1]),
    OpDivide([Id; 2]),
    OpDot([Id; 2]),
    OpExp([Id; 1]),
    OpEye(String),
    OpMaximum([Id; 2]),
    OpMinimum([Id; 2]),
    OpMultiply([Id; 2]),
    OpNegate([Id; 1]),
    OpOr([Id; 2]),
    OpReduceSum(String, [Id; 1]),
    OpReshape(String, [Id; 1]),
    OpReverse(String, [Id; 1]),
    OpShiftLeft([Id; 2]),
    OpShiftRightLogical([Id; 2]),
    OpSlice(String, [Id; 1]),
    OpSubtract([Id; 2]),
    OpXor([Id; 2]),
    OpTranspose(String, [Id; 1]),
    // other
    DetectedConst(String),
    Var(String),
}

impl TensorOp {
    pub fn num_children(&self) -> usize {
        match self {
            TensorOp::MvinSpad(..) => 1,
            TensorOp::MvoutSpad(..) => 1,
            TensorOp::MvoutSpadRelu(..) => 1,
            TensorOp::MvinAcc(..) => 1,
            TensorOp::MvoutAcc(..) => 1,
            TensorOp::MvoutAccRelu(..) => 1,
            TensorOp::MvinAccLow(..) => 1,
            TensorOp::MvinAccLowAdd(..) => 2,
            TensorOp::MvoutAccLow(..) => 1,
            TensorOp::MvoutAccLowRelu(..) => 1,
            TensorOp::Matmul8(..) => 2,
            TensorOp::Matmul32(..) => 2,
            TensorOp::Mac8(..) => 3,
            TensorOp::Mac32(..) => 3,
            TensorOp::OpAdd(..) => 2,
            TensorOp::OpBitcvt(..) => 1,
            TensorOp::OpBroadcast(..) => 1,
            TensorOp::OpClamp(..) => 3,
            TensorOp::OpConcat(..) => 2,
            TensorOp::OpConstant(..) => 0,
            TensorOp::OpConvert(..) => 1,
            TensorOp::OpCopy(..) => 1,
            TensorOp::OpDivide(..) => 2,
            TensorOp::OpDot(..) => 2,
            TensorOp::OpExp(..) => 1,
            TensorOp::OpEye(..) => 0,
            TensorOp::OpMaximum(..) => 2,
            TensorOp::OpMinimum(..) => 2,
            TensorOp::OpMultiply(..) => 2,
            TensorOp::OpNegate(..) => 1,
            TensorOp::OpOr(..) => 2,
            TensorOp::OpReduceSum(..) => 1,
            TensorOp::OpReshape(..) => 1,
            TensorOp::OpReverse(..) => 1,
            TensorOp::OpShiftLeft(..) => 2,
            TensorOp::OpShiftRightLogical(..) => 2,
            TensorOp::OpSlice(..) => 1,
            TensorOp::OpSubtract(..) => 2,
            TensorOp::OpXor(..) => 2,
            TensorOp::OpTranspose(..) => 1,
            TensorOp::DetectedConst(..) => 0,
            TensorOp::Var(..) => 0,
        }
    }

    pub fn is_detected_const(&self) -> bool {
        match self {
            TensorOp::DetectedConst(_) => true,
            _ => false,
        }
    }

    pub fn set_metadata(&mut self, metadata: Option<String>) {
        match self {
            TensorOp::MvinSpad(data, _) => *data = metadata.expect("MvinSpad needs metadata!"),
            TensorOp::MvoutSpad(data, _) => *data = metadata.expect("MvoutSpad needs metadata!"),
            TensorOp::MvoutSpadRelu(data, _) => *data = metadata.expect("MvoutSpadRelu needs metadata!"),
            TensorOp::MvinAcc(data, _) => *data = metadata.expect("MvinAcc needs metadata!"),
            TensorOp::MvoutAcc(data, _) => *data = metadata.expect("MvoutAcc needs metadata!"),
            TensorOp::MvoutAccRelu(data, _) => *data = metadata.expect("MvoutAccRelu needs metadata!"),
            TensorOp::MvinAccLow(data, _) => *data = metadata.expect("MvinAccLow needs metadata!"),
            TensorOp::MvinAccLowAdd(data, _) => *data = metadata.expect("MvinAccLowAdd needs metadata!"),
            TensorOp::MvoutAccLow(data, _) => *data = metadata.expect("MvoutAccLow needs metadata!"),
            TensorOp::MvoutAccLowRelu(data, _) => *data = metadata.expect("MvoutAccLowRelu needs metadata!"),
            TensorOp::Matmul8(data, _) => *data = metadata.expect("Matmul8 needs metadata!"),
            TensorOp::Matmul32(data, _) => *data = metadata.expect("Matmul32 needs metadata!"),
            TensorOp::Mac8(data, _) => *data = metadata.expect("Mac8 needs metadata!"),
            TensorOp::Mac32(data, _) => *data = metadata.expect("Mac32 needs metadata!"),
            TensorOp::OpBroadcast(data, _) => {
                *data = metadata.expect("OpBroadcast needs metadata!")
            }
            TensorOp::OpConcat(data, _) => *data = metadata.expect("OpConcat needs metadata!"),
            TensorOp::OpConstant(data) => *data = metadata.expect("OpConstant needs metadata!"),
            TensorOp::OpConvert(data, _) => *data = metadata.expect("OpConvert needs metadata!"),
            TensorOp::OpEye(data) => *data = metadata.expect("OpEye needs metadata!"),
            TensorOp::OpReduceSum(data, _) => {
                *data = metadata.expect("OpReduceSum needs metadata!")
            }
            TensorOp::OpReshape(data, _) => *data = metadata.expect("OpReshape needs metadata!"),
            TensorOp::OpReverse(data, _) => *data = metadata.expect("OpReverse needs metadata!"),
            TensorOp::OpSlice(data, _) => *data = metadata.expect("OpSlice needs metadata!"),
            TensorOp::OpTranspose(data, _) => {
                *data = metadata.expect("OpTranspose needs metadata!")
            }
            _ => (),
        }
    }
}

impl Language for TensorOp {
    type Discriminant = std::mem::Discriminant<Self>;

    fn discriminant(&self) -> Self::Discriminant {
        std::mem::discriminant(self)
    }

    // All variants have a fixed number of children, so if self and other are the same variant,
    // then they must have the same arity.
    fn matches(&self, other: &Self) -> bool {
        std::mem::discriminant(self) == std::mem::discriminant(other)
    }

    fn children(&self) -> &[Id] {
        match self {
            TensorOp::MvinSpad(_, ids) => LanguageChildren::as_slice(ids),
            TensorOp::MvoutSpad(_, ids) => LanguageChildren::as_slice(ids),
            TensorOp::MvoutSpadRelu(_, ids) => LanguageChildren::as_slice(ids),
            TensorOp::MvinAcc(_, ids) => LanguageChildren::as_slice(ids),
            TensorOp::MvoutAcc(_, ids) => LanguageChildren::as_slice(ids),
            TensorOp::MvoutAccRelu(_, ids) => LanguageChildren::as_slice(ids),
            TensorOp::MvinAccLow(_, ids) => LanguageChildren::as_slice(ids),
            TensorOp::MvinAccLowAdd(_, ids) => LanguageChildren::as_slice(ids),
            TensorOp::MvoutAccLow(_, ids) => LanguageChildren::as_slice(ids),
            TensorOp::MvoutAccLowRelu(_, ids) => LanguageChildren::as_slice(ids),
            TensorOp::Matmul8(_, ids) => LanguageChildren::as_slice(ids),
            TensorOp::Matmul32(_, ids) => LanguageChildren::as_slice(ids),
            TensorOp::Mac8(_, ids) => LanguageChildren::as_slice(ids),
            TensorOp::Mac32(_, ids) => LanguageChildren::as_slice(ids),
            TensorOp::OpAdd(ids) => LanguageChildren::as_slice(ids),
            TensorOp::OpBitcvt(ids) => LanguageChildren::as_slice(ids),
            TensorOp::OpBroadcast(_, ids) => LanguageChildren::as_slice(ids),
            TensorOp::OpClamp(ids) => LanguageChildren::as_slice(ids),
            TensorOp::OpConcat(_, ids) => LanguageChildren::as_slice(ids),
            TensorOp::OpConstant(_) => &[],
            TensorOp::OpConvert(_, ids) => LanguageChildren::as_slice(ids),
            TensorOp::OpCopy(ids) => LanguageChildren::as_slice(ids),
            TensorOp::OpDivide(ids) => LanguageChildren::as_slice(ids),
            TensorOp::OpDot(ids) => LanguageChildren::as_slice(ids),
            TensorOp::OpExp(ids) => LanguageChildren::as_slice(ids),
            TensorOp::OpEye(_) => &[],
            TensorOp::OpMaximum(ids) => LanguageChildren::as_slice(ids),
            TensorOp::OpMinimum(ids) => LanguageChildren::as_slice(ids),
            TensorOp::OpMultiply(ids) => LanguageChildren::as_slice(ids),
            TensorOp::OpNegate(ids) => LanguageChildren::as_slice(ids),
            TensorOp::OpOr(ids) => LanguageChildren::as_slice(ids),
            TensorOp::OpReduceSum(_, ids) => LanguageChildren::as_slice(ids),
            TensorOp::OpReshape(_, ids) => LanguageChildren::as_slice(ids),
            TensorOp::OpReverse(_, ids) => LanguageChildren::as_slice(ids),
            TensorOp::OpShiftLeft(ids) => LanguageChildren::as_slice(ids),
            TensorOp::OpShiftRightLogical(ids) => LanguageChildren::as_slice(ids),
            TensorOp::OpSlice(_, ids) => LanguageChildren::as_slice(ids),
            TensorOp::OpSubtract(ids) => LanguageChildren::as_slice(ids),
            TensorOp::OpXor(ids) => LanguageChildren::as_slice(ids),
            TensorOp::OpTranspose(_, ids) => LanguageChildren::as_slice(ids),
            TensorOp::DetectedConst(_) => &[],
            TensorOp::Var(_) => &[],
        }
    }

    fn children_mut(&mut self) -> &mut [Id] {
        match self {
            TensorOp::MvinSpad(_, ids) => LanguageChildren::as_mut_slice(ids),
            TensorOp::MvoutSpad(_, ids) => LanguageChildren::as_mut_slice(ids),
            TensorOp::MvoutSpadRelu(_, ids) => LanguageChildren::as_mut_slice(ids),
            TensorOp::MvinAcc(_, ids) => LanguageChildren::as_mut_slice(ids),
            TensorOp::MvoutAcc(_, ids) => LanguageChildren::as_mut_slice(ids),
            TensorOp::MvoutAccRelu(_, ids) => LanguageChildren::as_mut_slice(ids),
            TensorOp::MvinAccLow(_, ids) => LanguageChildren::as_mut_slice(ids),
            TensorOp::MvinAccLowAdd(_, ids) => LanguageChildren::as_mut_slice(ids),
            TensorOp::MvoutAccLow(_, ids) => LanguageChildren::as_mut_slice(ids),
            TensorOp::MvoutAccLowRelu(_, ids) => LanguageChildren::as_mut_slice(ids),
            TensorOp::Matmul8(_, ids) => LanguageChildren::as_mut_slice(ids),
            TensorOp::Matmul32(_, ids) => LanguageChildren::as_mut_slice(ids),
            TensorOp::Mac8(_, ids) => LanguageChildren::as_mut_slice(ids),
            TensorOp::Mac32(_, ids) => LanguageChildren::as_mut_slice(ids),
            TensorOp::OpAdd(ids) => LanguageChildren::as_mut_slice(ids),
            TensorOp::OpBitcvt(ids) => LanguageChildren::as_mut_slice(ids),
            TensorOp::OpBroadcast(_, ids) => LanguageChildren::as_mut_slice(ids),
            TensorOp::OpClamp(ids) => LanguageChildren::as_mut_slice(ids),
            TensorOp::OpConcat(_, ids) => LanguageChildren::as_mut_slice(ids),
            TensorOp::OpConstant(_) => &mut [],
            TensorOp::OpConvert(_, ids) => LanguageChildren::as_mut_slice(ids),
            TensorOp::OpCopy(ids) => LanguageChildren::as_mut_slice(ids),
            TensorOp::OpDivide(ids) => LanguageChildren::as_mut_slice(ids),
            TensorOp::OpDot(ids) => LanguageChildren::as_mut_slice(ids),
            TensorOp::OpExp(ids) => LanguageChildren::as_mut_slice(ids),
            TensorOp::OpEye(_) => &mut [],
            TensorOp::OpMaximum(ids) => LanguageChildren::as_mut_slice(ids),
            TensorOp::OpMinimum(ids) => LanguageChildren::as_mut_slice(ids),
            TensorOp::OpMultiply(ids) => LanguageChildren::as_mut_slice(ids),
            TensorOp::OpNegate(ids) => LanguageChildren::as_mut_slice(ids),
            TensorOp::OpOr(ids) => LanguageChildren::as_mut_slice(ids),
            TensorOp::OpReduceSum(_, ids) => LanguageChildren::as_mut_slice(ids),
            TensorOp::OpReshape(_, ids) => LanguageChildren::as_mut_slice(ids),
            TensorOp::OpReverse(_, ids) => LanguageChildren::as_mut_slice(ids),
            TensorOp::OpShiftLeft(ids) => LanguageChildren::as_mut_slice(ids),
            TensorOp::OpShiftRightLogical(ids) => LanguageChildren::as_mut_slice(ids),
            TensorOp::OpSlice(_, ids) => LanguageChildren::as_mut_slice(ids),
            TensorOp::OpSubtract(ids) => LanguageChildren::as_mut_slice(ids),
            TensorOp::OpXor(ids) => LanguageChildren::as_mut_slice(ids),
            TensorOp::OpTranspose(_, ids) => LanguageChildren::as_mut_slice(ids),
            TensorOp::DetectedConst(_) => &mut [],
            TensorOp::Var(_) => &mut [],
        }
    }
}

impl FromOp for TensorOp {
    type Error = FromOpError;

    // define_language picks the first variant where it is possible to parse data into type
    fn from_op(op: &str, children: Vec<Id>) -> Result<Self, Self::Error> {
        match op {
            op if op.split('_').next().unwrap() == "mvin-spad"
                && <[Id; 1] as LanguageChildren>::can_be_length(children.len()) =>
            {
                let data = op.split('_').last().unwrap();
                let children = <[Id; 1] as LanguageChildren>::from_vec(children);
                Ok(TensorOp::MvinSpad(data.to_string(), children))
            }
            op if op.split('_').next().unwrap() == "mvout-spad"
                && <[Id; 1] as LanguageChildren>::can_be_length(children.len()) =>
            {
                let data = op.split('_').last().unwrap();
                let children = <[Id; 1] as LanguageChildren>::from_vec(children);
                Ok(TensorOp::MvoutSpad(data.to_string(), children))
            }
            op if op.split('_').next().unwrap() == "mvout-spad-relu"
                && <[Id; 1] as LanguageChildren>::can_be_length(children.len()) =>
            {
                let data = op.split('_').last().unwrap();
                let children = <[Id; 1] as LanguageChildren>::from_vec(children);
                Ok(TensorOp::MvoutSpadRelu(data.to_string(), children))
            }
            op if op.split('_').next().unwrap() == "mvin-acc"
                && <[Id; 1] as LanguageChildren>::can_be_length(children.len()) =>
            {
                let data = op.split('_').last().unwrap();
                let children = <[Id; 1] as LanguageChildren>::from_vec(children);
                Ok(TensorOp::MvinAcc(data.to_string(), children))
            }
            op if op.split('_').next().unwrap() == "mvout-acc"
                && <[Id; 1] as LanguageChildren>::can_be_length(children.len()) =>
            {
                let data = op.split('_').last().unwrap();
                let children = <[Id; 1] as LanguageChildren>::from_vec(children);
                Ok(TensorOp::MvoutAcc(data.to_string(), children))
            }
            op if op.split('_').next().unwrap() == "mvout-acc-relu"
                && <[Id; 1] as LanguageChildren>::can_be_length(children.len()) =>
            {
                let data = op.split('_').last().unwrap();
                let children = <[Id; 1] as LanguageChildren>::from_vec(children);
                Ok(TensorOp::MvoutAccRelu(data.to_string(), children))
            }
            op if op.split('_').next().unwrap() == "mvin-acc-low"
                && <[Id; 1] as LanguageChildren>::can_be_length(children.len()) =>
            {
                let data = op.split('_').last().unwrap();
                let children = <[Id; 1] as LanguageChildren>::from_vec(children);
                Ok(TensorOp::MvinAccLow(data.to_string(), children))
            }
            op if op.split('_').next().unwrap() == "mvin-acc-low-add"
                && <[Id; 2] as LanguageChildren>::can_be_length(children.len()) =>
            {
                let data = op.split('_').last().unwrap();
                let children = <[Id; 2] as LanguageChildren>::from_vec(children);
                Ok(TensorOp::MvinAccLowAdd(data.to_string(), children))
            }
            op if op.split('_').next().unwrap() == "mvout-acc-low"
                && <[Id; 1] as LanguageChildren>::can_be_length(children.len()) =>
            {
                let data = op.split('_').last().unwrap();
                let children = <[Id; 1] as LanguageChildren>::from_vec(children);
                Ok(TensorOp::MvoutAccLow(data.to_string(), children))
            }
            op if op.split('_').next().unwrap() == "mvout-acc-low-relu"
                && <[Id; 1] as LanguageChildren>::can_be_length(children.len()) =>
            {
                let data = op.split('_').last().unwrap();
                let children = <[Id; 1] as LanguageChildren>::from_vec(children);
                Ok(TensorOp::MvoutAccLowRelu(data.to_string(), children))
            }
            op if op.split('_').next().unwrap() == "matmul8"
                && <[Id; 2] as LanguageChildren>::can_be_length(children.len()) =>
            {
                let data = op.split('_').last().unwrap();
                let children = <[Id; 2] as LanguageChildren>::from_vec(children);
                Ok(TensorOp::Matmul8(data.to_string(), children))
            }
            op if op.split('_').next().unwrap() == "matmul32"
                && <[Id; 2] as LanguageChildren>::can_be_length(children.len()) =>
            {
                let data = op.split('_').last().unwrap();
                let children = <[Id; 2] as LanguageChildren>::from_vec(children);
                Ok(TensorOp::Matmul32(data.to_string(), children))
            }
            op if op.split('_').next().unwrap() == "mac8"
                && <[Id; 3] as LanguageChildren>::can_be_length(children.len()) =>
            {
                let data = op.split('_').last().unwrap();
                let children = <[Id; 3] as LanguageChildren>::from_vec(children);
                Ok(TensorOp::Mac8(data.to_string(), children))
            }
            op if op.split('_').next().unwrap() == "mac32"
                && <[Id; 3] as LanguageChildren>::can_be_length(children.len()) =>
            {
                let data = op.split('_').last().unwrap();
                let children = <[Id; 3] as LanguageChildren>::from_vec(children);
                Ok(TensorOp::Mac32(data.to_string(), children))
            }
            op if op == "add" && <[Id; 2] as LanguageChildren>::can_be_length(children.len()) => {
                let children = <[Id; 2] as LanguageChildren>::from_vec(children);
                Ok(TensorOp::OpAdd(children))
            }
            op if op == "bitcvt"
                && <[Id; 1] as LanguageChildren>::can_be_length(children.len()) =>
            {
                let children = <[Id; 1] as LanguageChildren>::from_vec(children);
                Ok(TensorOp::OpBitcvt(children))
            }
            op if op.split('_').next().unwrap() == "broadcast"
                && <[Id; 1] as LanguageChildren>::can_be_length(children.len()) =>
            {
                let data = op.split('_').last().unwrap();
                let children = <[Id; 1] as LanguageChildren>::from_vec(children);
                Ok(TensorOp::OpBroadcast(data.to_string(), children))
            }
            op if op == "clamp"
                && <[Id; 3] as LanguageChildren>::can_be_length(children.len()) =>
            {
                let children = <[Id; 3] as LanguageChildren>::from_vec(children);
                Ok(TensorOp::OpClamp(children))
            }
            op if op.split('_').next().unwrap() == "concat"
                && <[Id; 2] as LanguageChildren>::can_be_length(children.len()) =>
            {
                let data = op.split('_').last().unwrap();
                let children = <[Id; 2] as LanguageChildren>::from_vec(children);
                Ok(TensorOp::OpConcat(data.to_string(), children))
            }
            op if op.split('_').next().unwrap() == "constant"
                && <[Id; 0] as LanguageChildren>::can_be_length(children.len()) =>
            {
                let data = op.split('_').last().unwrap();
                Ok(TensorOp::OpConstant(data.to_string()))
            }
            op if op.split('_').next().unwrap() == "convert"
                && <[Id; 1] as LanguageChildren>::can_be_length(children.len()) =>
            {
                let data = op.split('_').last().unwrap();
                let children = <[Id; 1] as LanguageChildren>::from_vec(children);
                Ok(TensorOp::OpConvert(data.to_string(), children))
            }
            op if op == "copy" && <[Id; 1] as LanguageChildren>::can_be_length(children.len()) => {
                let children = <[Id; 1] as LanguageChildren>::from_vec(children);
                Ok(TensorOp::OpCopy(children))
            }
            op if op == "divide"
                && <[Id; 2] as LanguageChildren>::can_be_length(children.len()) =>
            {
                let children = <[Id; 2] as LanguageChildren>::from_vec(children);
                Ok(TensorOp::OpDivide(children))
            }
            op if op == "dot" && <[Id; 2] as LanguageChildren>::can_be_length(children.len()) => {
                let children = <[Id; 2] as LanguageChildren>::from_vec(children);
                Ok(TensorOp::OpDot(children))
            }
            op if op == "exponential"
                && <[Id; 1] as LanguageChildren>::can_be_length(children.len()) =>
            {
                let children = <[Id; 1] as LanguageChildren>::from_vec(children);
                Ok(TensorOp::OpExp(children))
            }
            op if op.split('_').next().unwrap() == "eye"
                && <[Id; 0] as LanguageChildren>::can_be_length(children.len()) =>
            {
                let data = op.split('_').last().unwrap();
                Ok(TensorOp::OpEye(data.to_string()))
            }
            op if op == "maximum"
                && <[Id; 2] as LanguageChildren>::can_be_length(children.len()) =>
            {
                let children = <[Id; 2] as LanguageChildren>::from_vec(children);
                Ok(TensorOp::OpMaximum(children))
            }
            op if op == "minimum"
                && <[Id; 2] as LanguageChildren>::can_be_length(children.len()) =>
            {
                let children = <[Id; 2] as LanguageChildren>::from_vec(children);
                Ok(TensorOp::OpMinimum(children))
            }
            op if op == "multiply"
                && <[Id; 2] as LanguageChildren>::can_be_length(children.len()) =>
            {
                let children = <[Id; 2] as LanguageChildren>::from_vec(children);
                Ok(TensorOp::OpMultiply(children))
            }
            op if op == "negate"
                && <[Id; 1] as LanguageChildren>::can_be_length(children.len()) =>
            {
                let children = <[Id; 1] as LanguageChildren>::from_vec(children);
                Ok(TensorOp::OpNegate(children))
            }
            op if op == "or" && <[Id; 2] as LanguageChildren>::can_be_length(children.len()) => {
                let children = <[Id; 2] as LanguageChildren>::from_vec(children);
                Ok(TensorOp::OpOr(children))
            }
            op if op.split('_').next().unwrap() == "reduce"
                && <[Id; 1] as LanguageChildren>::can_be_length(children.len()) =>
            {
                let data = op.split('_').last().unwrap();
                let children = <[Id; 1] as LanguageChildren>::from_vec(children);
                Ok(TensorOp::OpReduceSum(data.to_string(), children))
            }
            op if op.split('_').next().unwrap() == "reshape"
                && <[Id; 1] as LanguageChildren>::can_be_length(children.len()) =>
            {
                let data = op.split('_').last().unwrap();
                let children = <[Id; 1] as LanguageChildren>::from_vec(children);
                Ok(TensorOp::OpReshape(data.to_string(), children))
            }
            op if op.split('_').next().unwrap() == "reverse"
                && <[Id; 1] as LanguageChildren>::can_be_length(children.len()) =>
            {
                let data = op.split('_').last().unwrap();
                let children = <[Id; 1] as LanguageChildren>::from_vec(children);
                Ok(TensorOp::OpReverse(data.to_string(), children))
            }
            op if op == "shift-left"
                && <[Id; 2] as LanguageChildren>::can_be_length(children.len()) =>
            {
                let children = <[Id; 2] as LanguageChildren>::from_vec(children);
                Ok(TensorOp::OpShiftLeft(children))
            }
            op if op == "shift-right-logical"
                && <[Id; 2] as LanguageChildren>::can_be_length(children.len()) =>
            {
                let children = <[Id; 2] as LanguageChildren>::from_vec(children);
                Ok(TensorOp::OpShiftRightLogical(children))
            }
            op if op.split('_').next().unwrap() == "slice"
                && <[Id; 1] as LanguageChildren>::can_be_length(children.len()) =>
            {
                let data = op.split('_').last().unwrap();
                let children = <[Id; 1] as LanguageChildren>::from_vec(children);
                Ok(TensorOp::OpSlice(data.to_string(), children))
            }
            op if op == "subtract"
                && <[Id; 2] as LanguageChildren>::can_be_length(children.len()) =>
            {
                let children = <[Id; 2] as LanguageChildren>::from_vec(children);
                Ok(TensorOp::OpSubtract(children))
            }
            op if op == "xor" && <[Id; 2] as LanguageChildren>::can_be_length(children.len()) => {
                let children = <[Id; 2] as LanguageChildren>::from_vec(children);
                Ok(TensorOp::OpXor(children))
            }
            op if op.split('_').next().unwrap() == "transpose"
                && <[Id; 1] as LanguageChildren>::can_be_length(children.len()) =>
            {
                let data = op.split('_').last().unwrap();
                let children = <[Id; 1] as LanguageChildren>::from_vec(children);
                Ok(TensorOp::OpTranspose(data.to_string(), children))
            }
            op if op.starts_with('?') && children.is_empty() => Ok(TensorOp::Var(op.to_string())),
            _ => Err(FromOpError::new(op, children)),
        }
    }
}

impl std::fmt::Display for TensorOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TensorOp::MvinSpad(data, _) => write!(f, "mvin_spad{{{}}}", data),
            TensorOp::MvoutSpad(data, _) => write!(f, "mvout_spad{{{}}}", data),
            TensorOp::MvoutSpadRelu(data, _) => write!(f, "mvout_spad_relu{{{}}}", data),
            TensorOp::MvinAcc(data, _) => write!(f, "mvin_acc{{{}}}", data),
            TensorOp::MvoutAcc(data, _) => write!(f, "mvout_acc{{{}}}", data),
            TensorOp::MvoutAccRelu(data, _) => write!(f, "mvout_acc_relu{{{}}}", data),
            TensorOp::MvinAccLow(data, _) => write!(f, "mvin_acc_low{{{}}}", data),
            TensorOp::MvinAccLowAdd(data, _) => write!(f, "mvin_acc_low_add{{{}}}", data),
            TensorOp::MvoutAccLow(data, _) => write!(f, "mvout_acc_low{{{}}}", data),
            TensorOp::MvoutAccLowRelu(data, _) => write!(f, "mvout_acc_low_relu{{{}}}", data),
            TensorOp::Matmul8(data, _) => write!(f, "matmul8{{{}}}", data),
            TensorOp::Matmul32(data, _) => write!(f, "matmul32{{{}}}", data),
            TensorOp::Mac8(data, _) => write!(f, "mac8{{{}}}", data),
            TensorOp::Mac32(data, _) => write!(f, "mac32{{{}}}", data),
            TensorOp::OpAdd(_) => write!(f, "add"),
            TensorOp::OpBitcvt(_) => write!(f, "bitcvt"),
            TensorOp::OpBroadcast(data, _) => write!(f, "broadcast{{{}}}", data),
            TensorOp::OpClamp(_) => write!(f, "clamp"),
            TensorOp::OpConcat(data, _) => write!(f, "concat{{{}}}", data),
            TensorOp::OpConstant(data) => write!(f, "constant{{{}}}", data),
            TensorOp::OpConvert(data, _) => write!(f, "convert{{{}}}", data),
            TensorOp::OpCopy(_) => write!(f, "copy"),
            TensorOp::OpDivide(_) => write!(f, "divide"),
            TensorOp::OpDot(_) => write!(f, "dot"),
            TensorOp::OpExp(_) => write!(f, "exponential"),
            TensorOp::OpEye(data) => write!(f, "eye{{{}}}", data),
            TensorOp::OpMaximum(_) => write!(f, "maximum"),
            TensorOp::OpMinimum(_) => write!(f, "minimum"),
            TensorOp::OpMultiply(_) => write!(f, "multiply"),
            TensorOp::OpNegate(_) => write!(f, "negate"),
            TensorOp::OpOr(_) => write!(f, "or"),
            TensorOp::OpReduceSum(data, _) => write!(f, "reduce{{{}}}", data),
            TensorOp::OpReshape(data, _) => write!(f, "reshape{{{}}}", data),
            TensorOp::OpReverse(data, _) => write!(f, "reverse{{{}}}", data),
            TensorOp::OpShiftLeft(_) => write!(f, "shift_left"),
            TensorOp::OpShiftRightLogical(_) => write!(f, "shift_right_logical"),
            TensorOp::OpSlice(data, _) => write!(f, "slice{{{}}}", data),
            TensorOp::OpSubtract(_) => write!(f, "subtract"),
            TensorOp::OpXor(_) => write!(f, "xor"),
            TensorOp::OpTranspose(data, _) => write!(f, "transpose{{{}}}", data),
            TensorOp::DetectedConst(id) => write!(f, "DCC{{{}}}", id),
            TensorOp::Var(v) => write!(f, "Var{{{}}}", v),
        }
    }
}

// E-class metadata
#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub shape: Vec<i32>,
    pub dtype: Dtype,
    pub is_const: bool,
}

impl Default for TensorInfo {
    fn default() -> Self {
        TensorInfo {
            shape: vec![],
            dtype: Dtype::U8,
            is_const: false,
        }
    }
}

impl PartialEq for TensorInfo {
    fn eq(&self, other: &Self) -> bool {
        self.shape == other.shape && self.dtype == other.dtype
    }
}

impl Analysis<TensorOp> for TensorInfo {
    type Data = TensorInfo;

    fn make(_egraph: &mut EGraph<TensorOp, Self>, enode: &TensorOp) -> Self::Data {
        let mut data = TensorInfo::default();
        data.is_const = match enode {
            TensorOp::DetectedConst(_) => true,
            _ => false,
        };
        data
    }

    // TODO: ensure that the two eclasses have the same shape
    fn merge(&mut self, a: &mut Self::Data, b: Self::Data) -> DidMerge {
        let x = a.is_const;
        a.is_const |= b.is_const;
        DidMerge(a.is_const != x, false)
    }
}
