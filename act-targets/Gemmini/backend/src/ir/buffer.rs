use crate::ir::egraph::TensorOp;

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum Buffer {
    HBM,
    SPAD,
    ACC,
    ANY,
}

impl std::fmt::Display for Buffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Buffer::HBM => write!(f, "HBM"),
            Buffer::SPAD => write!(f, "SPAD"),
            Buffer::ACC => write!(f, "ACC"),
            Buffer::ANY => panic!("Buffer::ANY does not have a string representation"),
        }
    }
}

// Return buffer assignment for an instruction enode.
// Returns Some(vec![out_buf, in_buf1, in_buf2, ...]) or None if not applicable.
pub fn buffer_assignment(en: &TensorOp) -> Option<Vec<Buffer>> {
    match en {
        TensorOp::MvinSpad(_, _) => Some(vec![Buffer::SPAD, Buffer::HBM]),
        TensorOp::MvoutSpad(_, _) => Some(vec![Buffer::HBM, Buffer::SPAD]),
        TensorOp::MvoutSpadRelu(_, _) => Some(vec![Buffer::HBM, Buffer::SPAD]),
        TensorOp::MvinAcc(_, _) => Some(vec![Buffer::ACC, Buffer::HBM]),
        TensorOp::MvoutAcc(_, _) => Some(vec![Buffer::HBM, Buffer::ACC]),
        TensorOp::MvoutAccRelu(_, _) => Some(vec![Buffer::HBM, Buffer::ACC]),
        TensorOp::MvinAccLow(_, _) => Some(vec![Buffer::ACC, Buffer::HBM]),
        TensorOp::MvinAccLowAdd(_, _) => Some(vec![Buffer::ACC, Buffer::ACC, Buffer::HBM]),
        TensorOp::MvoutAccLow(_, _) => Some(vec![Buffer::HBM, Buffer::ACC]),
        TensorOp::MvoutAccLowRelu(_, _) => Some(vec![Buffer::HBM, Buffer::ACC]),
        TensorOp::Matmul8(_, _) => Some(vec![Buffer::SPAD, Buffer::SPAD, Buffer::SPAD]),
        TensorOp::Matmul32(_, _) => Some(vec![Buffer::ACC, Buffer::SPAD, Buffer::SPAD]),
        TensorOp::Mac8(_, _) => Some(vec![Buffer::SPAD, Buffer::SPAD, Buffer::SPAD, Buffer::SPAD]),
        TensorOp::Mac32(_, _) => Some(vec![Buffer::ACC, Buffer::SPAD, Buffer::SPAD, Buffer::ACC]),
        TensorOp::OpSlice(_, _) => Some(vec![Buffer::ANY, Buffer::ANY]),
        TensorOp::OpConcat(_, _) => Some(vec![Buffer::ANY, Buffer::ANY, Buffer::ANY]),
        TensorOp::DetectedConst(_) => Some(vec![Buffer::HBM]),
        TensorOp::Var(_) => Some(vec![Buffer::HBM]),
        _ => None,
    }
}
