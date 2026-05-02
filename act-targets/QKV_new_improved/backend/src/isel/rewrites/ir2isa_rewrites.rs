use egg::{EGraph, Id};

use crate::ir::dtype::Dtype;
use crate::ir::egraph::{TensorInfo, TensorOp};

pub fn precond_load_01(egraph: &EGraph<TensorOp, TensorInfo>, lhs_eclasses: &Vec<Id>) -> bool {
    assert_eq!(lhs_eclasses.len(), 4);
    let lhs_metadata: Vec<TensorInfo> = lhs_eclasses.iter().map(|id| egraph[*id].data.clone()).collect();
    let n = lhs_metadata[3].shape[0];

    if lhs_metadata[0] != (TensorInfo { shape: vec![n*128], dtype: Dtype::U8, is_const: false, }) {
        return false;
    }
    if lhs_metadata[1] != (TensorInfo { shape: vec![n*128], dtype: Dtype::U8, is_const: false, }) {
        return false;
    }
    if lhs_metadata[2] != (TensorInfo { shape: vec![n, 64, 2], dtype: Dtype::U8, is_const: false, }) {
        return false;
    }
    if lhs_metadata[3] != (TensorInfo { shape: vec![n, 64], dtype: Dtype::BF16, is_const: false, }) {
        return false;
    }
    true
}


pub fn metadata_load_01(
    egraph: &EGraph<TensorOp, TensorInfo>,
    lhs_eclasses: &Vec<Id>,
    _lhs_enodes: &Vec<Option<TensorOp>>,
) -> Vec<Option<String>> {
    assert_eq!(lhs_eclasses.len(), 4);
    let lhs_metadata: Vec<TensorInfo> = lhs_eclasses.iter().map(|id| egraph[*id].data.clone()).collect();
    let n = lhs_metadata[3].shape[0];

    let mut rhs_metadata = vec![None; 3];
    let isa_idx = rhs_metadata.len() - 2;
    rhs_metadata[isa_idx] = Some(n.to_string());
    rhs_metadata
}


pub fn set_shapes_load_01(
    egraph: &mut EGraph<TensorOp, TensorInfo>,
    lhs_eclasses: &Vec<Id>,
    rhs_eclasses: &Vec<Id>,
) {
    assert_eq!(lhs_eclasses.len(), 4);
    assert_eq!(rhs_eclasses.len(), 3);
    let lhs_metadata: Vec<TensorInfo> = lhs_eclasses.iter().map(|id| egraph[*id].data.clone()).collect();
    let n = lhs_metadata[3].shape[0];


    let info = TensorInfo { shape: vec![n, 64], dtype: Dtype::BF16, is_const: false, };
    let isa_idx = rhs_eclasses.len() - 2;
    egraph.set_analysis_data(rhs_eclasses[isa_idx], info.clone());
    egraph.set_analysis_data(*rhs_eclasses.last().unwrap(), info);

}


pub fn precond_load_03(egraph: &EGraph<TensorOp, TensorInfo>, lhs_eclasses: &Vec<Id>) -> bool {
    assert_eq!(lhs_eclasses.len(), 4);
    let lhs_metadata: Vec<TensorInfo> = lhs_eclasses.iter().map(|id| egraph[*id].data.clone()).collect();
    let n = lhs_metadata[3].shape[0];

    if lhs_metadata[0] != (TensorInfo { shape: vec![n*128], dtype: Dtype::U8, is_const: false, }) {
        return false;
    }
    if lhs_metadata[1] != (TensorInfo { shape: vec![n*128], dtype: Dtype::U8, is_const: false, }) {
        return false;
    }
    if lhs_metadata[2] != (TensorInfo { shape: vec![n, 64, 2], dtype: Dtype::U8, is_const: false, }) {
        return false;
    }
    if lhs_metadata[3] != (TensorInfo { shape: vec![n, 64], dtype: Dtype::BF16, is_const: false, }) {
        return false;
    }
    true
}


pub fn metadata_load_03(
    egraph: &EGraph<TensorOp, TensorInfo>,
    lhs_eclasses: &Vec<Id>,
    _lhs_enodes: &Vec<Option<TensorOp>>,
) -> Vec<Option<String>> {
    assert_eq!(lhs_eclasses.len(), 4);
    let lhs_metadata: Vec<TensorInfo> = lhs_eclasses.iter().map(|id| egraph[*id].data.clone()).collect();
    let n = lhs_metadata[3].shape[0];

    let mut rhs_metadata = vec![None; 3];
    let isa_idx = rhs_metadata.len() - 2;
    rhs_metadata[isa_idx] = Some(n.to_string());
    rhs_metadata
}


pub fn set_shapes_load_03(
    egraph: &mut EGraph<TensorOp, TensorInfo>,
    lhs_eclasses: &Vec<Id>,
    rhs_eclasses: &Vec<Id>,
) {
    assert_eq!(lhs_eclasses.len(), 4);
    assert_eq!(rhs_eclasses.len(), 3);
    let lhs_metadata: Vec<TensorInfo> = lhs_eclasses.iter().map(|id| egraph[*id].data.clone()).collect();
    let n = lhs_metadata[3].shape[0];


    let info = TensorInfo { shape: vec![n, 64], dtype: Dtype::BF16, is_const: false, };
    let isa_idx = rhs_eclasses.len() - 2;
    egraph.set_analysis_data(rhs_eclasses[isa_idx], info.clone());
    egraph.set_analysis_data(*rhs_eclasses.last().unwrap(), info);

}


pub fn precond_store_10(egraph: &EGraph<TensorOp, TensorInfo>, lhs_eclasses: &Vec<Id>) -> bool {
    assert_eq!(lhs_eclasses.len(), 4);
    let lhs_metadata: Vec<TensorInfo> = lhs_eclasses.iter().map(|id| egraph[*id].data.clone()).collect();
    let n = lhs_metadata[3].shape[0] / 128;

    if lhs_metadata[0] != (TensorInfo { shape: vec![n, 64], dtype: Dtype::BF16, is_const: false, }) {
        return false;
    }
    if lhs_metadata[1] != (TensorInfo { shape: vec![n, 64], dtype: Dtype::BF16, is_const: false, }) {
        return false;
    }
    if lhs_metadata[2] != (TensorInfo { shape: vec![n, 64, 2], dtype: Dtype::U8, is_const: false, }) {
        return false;
    }
    if lhs_metadata[3] != (TensorInfo { shape: vec![n*128], dtype: Dtype::U8, is_const: false, }) {
        return false;
    }
    true
}


pub fn metadata_store_10(
    egraph: &EGraph<TensorOp, TensorInfo>,
    lhs_eclasses: &Vec<Id>,
    _lhs_enodes: &Vec<Option<TensorOp>>,
) -> Vec<Option<String>> {
    assert_eq!(lhs_eclasses.len(), 4);
    let lhs_metadata: Vec<TensorInfo> = lhs_eclasses.iter().map(|id| egraph[*id].data.clone()).collect();
    let n = lhs_metadata[3].shape[0] / 128;

    let mut rhs_metadata = vec![None; 3];
    let isa_idx = rhs_metadata.len() - 2;
    rhs_metadata[isa_idx] = Some(n.to_string());
    rhs_metadata
}


pub fn set_shapes_store_10(
    egraph: &mut EGraph<TensorOp, TensorInfo>,
    lhs_eclasses: &Vec<Id>,
    rhs_eclasses: &Vec<Id>,
) {
    assert_eq!(lhs_eclasses.len(), 4);
    assert_eq!(rhs_eclasses.len(), 3);
    let lhs_metadata: Vec<TensorInfo> = lhs_eclasses.iter().map(|id| egraph[*id].data.clone()).collect();
    let n = lhs_metadata[3].shape[0] / 128;


    let info = TensorInfo { shape: vec![n*128], dtype: Dtype::U8, is_const: false, };
    let isa_idx = rhs_eclasses.len() - 2;
    egraph.set_analysis_data(rhs_eclasses[isa_idx], info.clone());
    egraph.set_analysis_data(*rhs_eclasses.last().unwrap(), info);

}


pub fn precond_store_30(egraph: &EGraph<TensorOp, TensorInfo>, lhs_eclasses: &Vec<Id>) -> bool {
    assert_eq!(lhs_eclasses.len(), 4);
    let lhs_metadata: Vec<TensorInfo> = lhs_eclasses.iter().map(|id| egraph[*id].data.clone()).collect();
    let n = lhs_metadata[3].shape[0] / 128;

    if lhs_metadata[0] != (TensorInfo { shape: vec![n, 64], dtype: Dtype::BF16, is_const: false, }) {
        return false;
    }
    if lhs_metadata[1] != (TensorInfo { shape: vec![n, 64], dtype: Dtype::BF16, is_const: false, }) {
        return false;
    }
    if lhs_metadata[2] != (TensorInfo { shape: vec![n, 64, 2], dtype: Dtype::U8, is_const: false, }) {
        return false;
    }
    if lhs_metadata[3] != (TensorInfo { shape: vec![n*128], dtype: Dtype::U8, is_const: false, }) {
        return false;
    }
    true
}


pub fn metadata_store_30(
    egraph: &EGraph<TensorOp, TensorInfo>,
    lhs_eclasses: &Vec<Id>,
    _lhs_enodes: &Vec<Option<TensorOp>>,
) -> Vec<Option<String>> {
    assert_eq!(lhs_eclasses.len(), 4);
    let lhs_metadata: Vec<TensorInfo> = lhs_eclasses.iter().map(|id| egraph[*id].data.clone()).collect();
    let n = lhs_metadata[3].shape[0] / 128;

    let mut rhs_metadata = vec![None; 3];
    let isa_idx = rhs_metadata.len() - 2;
    rhs_metadata[isa_idx] = Some(n.to_string());
    rhs_metadata
}


pub fn set_shapes_store_30(
    egraph: &mut EGraph<TensorOp, TensorInfo>,
    lhs_eclasses: &Vec<Id>,
    rhs_eclasses: &Vec<Id>,
) {
    assert_eq!(lhs_eclasses.len(), 4);
    assert_eq!(rhs_eclasses.len(), 3);
    let lhs_metadata: Vec<TensorInfo> = lhs_eclasses.iter().map(|id| egraph[*id].data.clone()).collect();
    let n = lhs_metadata[3].shape[0] / 128;


    let info = TensorInfo { shape: vec![n*128], dtype: Dtype::U8, is_const: false, };
    let isa_idx = rhs_eclasses.len() - 2;
    egraph.set_analysis_data(rhs_eclasses[isa_idx], info.clone());
    egraph.set_analysis_data(*rhs_eclasses.last().unwrap(), info);

}


pub fn precond_transpose_13(egraph: &EGraph<TensorOp, TensorInfo>, lhs_eclasses: &Vec<Id>) -> bool {
    assert_eq!(lhs_eclasses.len(), 4);
    let lhs_metadata: Vec<TensorInfo> = lhs_eclasses.iter().map(|id| egraph[*id].data.clone()).collect();

    if lhs_metadata[0] != (TensorInfo { shape: vec![64, 64], dtype: Dtype::BF16, is_const: false, }) {
        return false;
    }
    if lhs_metadata[1] != (TensorInfo { shape: vec![64, 64], dtype: Dtype::BF16, is_const: false, }) {
        return false;
    }
    if lhs_metadata[2] != (TensorInfo { shape: vec![64, 64], dtype: Dtype::BF16, is_const: false, }) {
        return false;
    }
    if lhs_metadata[3] != (TensorInfo { shape: vec![64, 64], dtype: Dtype::BF16, is_const: false, }) {
        return false;
    }
    true
}


pub fn metadata_transpose_13(
    egraph: &EGraph<TensorOp, TensorInfo>,
    lhs_eclasses: &Vec<Id>,
    _lhs_enodes: &Vec<Option<TensorOp>>,
) -> Vec<Option<String>> {
    assert_eq!(lhs_eclasses.len(), 4);
    let _lhs_metadata: Vec<TensorInfo> = lhs_eclasses.iter().map(|id| egraph[*id].data.clone()).collect();

    let rhs_metadata = vec![None; 3];
    rhs_metadata
}


pub fn set_shapes_transpose_13(
    egraph: &mut EGraph<TensorOp, TensorInfo>,
    lhs_eclasses: &Vec<Id>,
    rhs_eclasses: &Vec<Id>,
) {
    assert_eq!(lhs_eclasses.len(), 4);
    assert_eq!(rhs_eclasses.len(), 3);
    let _lhs_metadata: Vec<TensorInfo> = lhs_eclasses.iter().map(|id| egraph[*id].data.clone()).collect();


    let info = TensorInfo { shape: vec![64, 64], dtype: Dtype::BF16, is_const: false, };
    let isa_idx = rhs_eclasses.len() - 2;
    egraph.set_analysis_data(rhs_eclasses[isa_idx], info.clone());
    egraph.set_analysis_data(*rhs_eclasses.last().unwrap(), info);

}


pub fn precond_mov_21(egraph: &EGraph<TensorOp, TensorInfo>, lhs_eclasses: &Vec<Id>) -> bool {
    assert_eq!(lhs_eclasses.len(), 3);
    let lhs_metadata: Vec<TensorInfo> = lhs_eclasses.iter().map(|id| egraph[*id].data.clone()).collect();
    let n = lhs_metadata[2].shape[0];

    if lhs_metadata[0] != (TensorInfo { shape: vec![n, 64], dtype: Dtype::BF16, is_const: false, }) {
        return false;
    }
    if lhs_metadata[1] != (TensorInfo { shape: vec![n, 64], dtype: Dtype::BF16, is_const: false, }) {
        return false;
    }
    if lhs_metadata[2] != (TensorInfo { shape: vec![n, 64], dtype: Dtype::BF16, is_const: false, }) {
        return false;
    }
    true
}


pub fn metadata_mov_21(
    egraph: &EGraph<TensorOp, TensorInfo>,
    lhs_eclasses: &Vec<Id>,
    _lhs_enodes: &Vec<Option<TensorOp>>,
) -> Vec<Option<String>> {
    assert_eq!(lhs_eclasses.len(), 3);
    let lhs_metadata: Vec<TensorInfo> = lhs_eclasses.iter().map(|id| egraph[*id].data.clone()).collect();
    let n = lhs_metadata[2].shape[0];

    let mut rhs_metadata = vec![None; 3];
    let isa_idx = rhs_metadata.len() - 2;
    rhs_metadata[isa_idx] = Some(n.to_string());
    rhs_metadata
}


pub fn set_shapes_mov_21(
    egraph: &mut EGraph<TensorOp, TensorInfo>,
    lhs_eclasses: &Vec<Id>,
    rhs_eclasses: &Vec<Id>,
) {
    assert_eq!(lhs_eclasses.len(), 3);
    assert_eq!(rhs_eclasses.len(), 3);
    let lhs_metadata: Vec<TensorInfo> = lhs_eclasses.iter().map(|id| egraph[*id].data.clone()).collect();
    let n = lhs_metadata[2].shape[0];


    let info = TensorInfo { shape: vec![n, 64], dtype: Dtype::BF16, is_const: false, };
    let isa_idx = rhs_eclasses.len() - 2;
    egraph.set_analysis_data(rhs_eclasses[isa_idx], info.clone());
    egraph.set_analysis_data(*rhs_eclasses.last().unwrap(), info);

}


pub fn precond_mov_23(egraph: &EGraph<TensorOp, TensorInfo>, lhs_eclasses: &Vec<Id>) -> bool {
    assert_eq!(lhs_eclasses.len(), 3);
    let lhs_metadata: Vec<TensorInfo> = lhs_eclasses.iter().map(|id| egraph[*id].data.clone()).collect();
    let n = lhs_metadata[2].shape[0];

    if lhs_metadata[0] != (TensorInfo { shape: vec![n, 64], dtype: Dtype::BF16, is_const: false, }) {
        return false;
    }
    if lhs_metadata[1] != (TensorInfo { shape: vec![n, 64], dtype: Dtype::BF16, is_const: false, }) {
        return false;
    }
    if lhs_metadata[2] != (TensorInfo { shape: vec![n, 64], dtype: Dtype::BF16, is_const: false, }) {
        return false;
    }
    true
}


pub fn metadata_mov_23(
    egraph: &EGraph<TensorOp, TensorInfo>,
    lhs_eclasses: &Vec<Id>,
    _lhs_enodes: &Vec<Option<TensorOp>>,
) -> Vec<Option<String>> {
    assert_eq!(lhs_eclasses.len(), 3);
    let lhs_metadata: Vec<TensorInfo> = lhs_eclasses.iter().map(|id| egraph[*id].data.clone()).collect();
    let n = lhs_metadata[2].shape[0];

    let mut rhs_metadata = vec![None; 3];
    let isa_idx = rhs_metadata.len() - 2;
    rhs_metadata[isa_idx] = Some(n.to_string());
    rhs_metadata
}


pub fn set_shapes_mov_23(
    egraph: &mut EGraph<TensorOp, TensorInfo>,
    lhs_eclasses: &Vec<Id>,
    rhs_eclasses: &Vec<Id>,
) {
    assert_eq!(lhs_eclasses.len(), 3);
    assert_eq!(rhs_eclasses.len(), 3);
    let lhs_metadata: Vec<TensorInfo> = lhs_eclasses.iter().map(|id| egraph[*id].data.clone()).collect();
    let n = lhs_metadata[2].shape[0];


    let info = TensorInfo { shape: vec![n, 64], dtype: Dtype::BF16, is_const: false, };
    let isa_idx = rhs_eclasses.len() - 2;
    egraph.set_analysis_data(rhs_eclasses[isa_idx], info.clone());
    egraph.set_analysis_data(*rhs_eclasses.last().unwrap(), info);

}


pub fn precond_gemm_33(egraph: &EGraph<TensorOp, TensorInfo>, lhs_eclasses: &Vec<Id>) -> bool {
    assert_eq!(lhs_eclasses.len(), 5);
    let lhs_metadata: Vec<TensorInfo> = lhs_eclasses.iter().map(|id| egraph[*id].data.clone()).collect();

    if lhs_metadata[0] != (TensorInfo { shape: vec![64, 64], dtype: Dtype::BF16, is_const: false, }) {
        return false;
    }
    if lhs_metadata[1] != (TensorInfo { shape: vec![64, 64], dtype: Dtype::BF16, is_const: false, }) {
        return false;
    }
    if lhs_metadata[2] != (TensorInfo { shape: vec![64, 64], dtype: Dtype::BF16, is_const: false, }) {
        return false;
    }
    if lhs_metadata[3] != (TensorInfo { shape: vec![64, 64], dtype: Dtype::BF16, is_const: false, }) {
        return false;
    }
    if lhs_metadata[4] != (TensorInfo { shape: vec![64, 64], dtype: Dtype::BF16, is_const: false, }) {
        return false;
    }
    true
}


pub fn metadata_gemm_33(
    egraph: &EGraph<TensorOp, TensorInfo>,
    lhs_eclasses: &Vec<Id>,
    _lhs_enodes: &Vec<Option<TensorOp>>,
) -> Vec<Option<String>> {
    assert_eq!(lhs_eclasses.len(), 5);
    let _lhs_metadata: Vec<TensorInfo> = lhs_eclasses.iter().map(|id| egraph[*id].data.clone()).collect();

    let rhs_metadata = vec![None; 4];
    rhs_metadata
}


pub fn set_shapes_gemm_33(
    egraph: &mut EGraph<TensorOp, TensorInfo>,
    lhs_eclasses: &Vec<Id>,
    rhs_eclasses: &Vec<Id>,
) {
    assert_eq!(lhs_eclasses.len(), 5);
    assert_eq!(rhs_eclasses.len(), 4);
    let _lhs_metadata: Vec<TensorInfo> = lhs_eclasses.iter().map(|id| egraph[*id].data.clone()).collect();


    let info = TensorInfo { shape: vec![64, 64], dtype: Dtype::BF16, is_const: false, };
    let isa_idx = rhs_eclasses.len() - 2;
    egraph.set_analysis_data(rhs_eclasses[isa_idx], info.clone());
    egraph.set_analysis_data(*rhs_eclasses.last().unwrap(), info);

}


pub fn precond_gemm_13(egraph: &EGraph<TensorOp, TensorInfo>, lhs_eclasses: &Vec<Id>) -> bool {
    assert_eq!(lhs_eclasses.len(), 5);
    let lhs_metadata: Vec<TensorInfo> = lhs_eclasses.iter().map(|id| egraph[*id].data.clone()).collect();

    if lhs_metadata[0] != (TensorInfo { shape: vec![64, 64], dtype: Dtype::BF16, is_const: false, }) {
        return false;
    }
    if lhs_metadata[1] != (TensorInfo { shape: vec![64, 64], dtype: Dtype::BF16, is_const: false, }) {
        return false;
    }
    if lhs_metadata[2] != (TensorInfo { shape: vec![64, 64], dtype: Dtype::BF16, is_const: false, }) {
        return false;
    }
    if lhs_metadata[3] != (TensorInfo { shape: vec![64, 64], dtype: Dtype::BF16, is_const: false, }) {
        return false;
    }
    if lhs_metadata[4] != (TensorInfo { shape: vec![64, 64], dtype: Dtype::BF16, is_const: false, }) {
        return false;
    }
    true
}


pub fn metadata_gemm_13(
    egraph: &EGraph<TensorOp, TensorInfo>,
    lhs_eclasses: &Vec<Id>,
    _lhs_enodes: &Vec<Option<TensorOp>>,
) -> Vec<Option<String>> {
    assert_eq!(lhs_eclasses.len(), 5);
    let _lhs_metadata: Vec<TensorInfo> = lhs_eclasses.iter().map(|id| egraph[*id].data.clone()).collect();

    let rhs_metadata = vec![None; 4];
    rhs_metadata
}


pub fn set_shapes_gemm_13(
    egraph: &mut EGraph<TensorOp, TensorInfo>,
    lhs_eclasses: &Vec<Id>,
    rhs_eclasses: &Vec<Id>,
) {
    assert_eq!(lhs_eclasses.len(), 5);
    assert_eq!(rhs_eclasses.len(), 4);
    let _lhs_metadata: Vec<TensorInfo> = lhs_eclasses.iter().map(|id| egraph[*id].data.clone()).collect();


    let info = TensorInfo { shape: vec![64, 64], dtype: Dtype::BF16, is_const: false, };
    let isa_idx = rhs_eclasses.len() - 2;
    egraph.set_analysis_data(rhs_eclasses[isa_idx], info.clone());
    egraph.set_analysis_data(*rhs_eclasses.last().unwrap(), info);

}


pub fn precond_softmax(egraph: &EGraph<TensorOp, TensorInfo>, lhs_eclasses: &Vec<Id>) -> bool {
    assert_eq!(lhs_eclasses.len(), 9);
    let lhs_metadata: Vec<TensorInfo> = lhs_eclasses.iter().map(|id| egraph[*id].data.clone()).collect();
    let n = lhs_metadata[8].shape[0];

    if lhs_metadata[0] != (TensorInfo { shape: vec![n, 64], dtype: Dtype::BF16, is_const: false, }) {
        return false;
    }
    if lhs_metadata[1] != (TensorInfo { shape: vec![n, 64], dtype: Dtype::BF16, is_const: false, }) {
        return false;
    }
    if lhs_metadata[2] != (TensorInfo { shape: vec![n, 64], dtype: Dtype::BF16, is_const: false, }) {
        return false;
    }
    if lhs_metadata[3] != (TensorInfo { shape: vec![n, 64], dtype: Dtype::BF16, is_const: false, }) {
        return false;
    }
    if lhs_metadata[4] != (TensorInfo { shape: vec![n, 64], dtype: Dtype::BF16, is_const: false, }) {
        return false;
    }
    if lhs_metadata[5] != (TensorInfo { shape: vec![n, 64], dtype: Dtype::BF16, is_const: false, }) {
        return false;
    }
    if lhs_metadata[6] != (TensorInfo { shape: vec![n], dtype: Dtype::BF16, is_const: false, }) {
        return false;
    }
    if lhs_metadata[7] != (TensorInfo { shape: vec![n, 64], dtype: Dtype::BF16, is_const: false, }) {
        return false;
    }
    if lhs_metadata[8] != (TensorInfo { shape: vec![n, 64], dtype: Dtype::BF16, is_const: false, }) {
        return false;
    }
    true
}


pub fn metadata_softmax(
    egraph: &EGraph<TensorOp, TensorInfo>,
    lhs_eclasses: &Vec<Id>,
    _lhs_enodes: &Vec<Option<TensorOp>>,
) -> Vec<Option<String>> {
    assert_eq!(lhs_eclasses.len(), 9);
    let lhs_metadata: Vec<TensorInfo> = lhs_eclasses.iter().map(|id| egraph[*id].data.clone()).collect();
    let n = lhs_metadata[8].shape[0];

    let mut rhs_metadata = vec![None; 3];
    let isa_idx = rhs_metadata.len() - 2;
    rhs_metadata[isa_idx] = Some(n.to_string());
    rhs_metadata
}


pub fn set_shapes_softmax(
    egraph: &mut EGraph<TensorOp, TensorInfo>,
    lhs_eclasses: &Vec<Id>,
    rhs_eclasses: &Vec<Id>,
) {
    assert_eq!(lhs_eclasses.len(), 9);
    assert_eq!(rhs_eclasses.len(), 3);
    let lhs_metadata: Vec<TensorInfo> = lhs_eclasses.iter().map(|id| egraph[*id].data.clone()).collect();
    let n = lhs_metadata[8].shape[0];


    let info = TensorInfo { shape: vec![n, 64], dtype: Dtype::BF16, is_const: false, };
    let isa_idx = rhs_eclasses.len() - 2;
    egraph.set_analysis_data(rhs_eclasses[isa_idx], info.clone());
    egraph.set_analysis_data(*rhs_eclasses.last().unwrap(), info);

}



