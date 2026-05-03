use egg::{EGraph, Id};

use crate::ir::dtype::Dtype;
use crate::ir::egraph::{TensorInfo, TensorOp};

pub fn precond_mvin_spad(egraph: &EGraph<TensorOp, TensorInfo>, lhs_eclasses: &Vec<Id>) -> bool {
    assert_eq!(lhs_eclasses.len(), 6);
    let lhs_metadata: Vec<TensorInfo> = lhs_eclasses.iter().map(|id| egraph[*id].data.clone()).collect();
    let shape_1 = &lhs_metadata[1].shape;
    if shape_1.len() <= 0 { return false; }
    let rows = shape_1[0];
    let shape_1 = &lhs_metadata[1].shape;
    if shape_1.len() <= 1 { return false; }
    let cols = shape_1[1];
    let scale = {
        let ec = lhs_eclasses[3];
        let mut val = 0i32;
        for en in egraph[ec].nodes.iter() {
            if let TensorOp::OpConstant(v) = en {
                let num_str = v.split(':').next().unwrap_or(v);
                val = num_str.parse::<i32>().unwrap_or(0);
                break;
            }
        }
        val
    };

    if lhs_metadata[0].shape != vec![rows*cols] || lhs_metadata[0].dtype != Dtype::U8 {
        return false;
    }
    if lhs_metadata[1].shape != vec![rows, cols] || lhs_metadata[1].dtype != Dtype::U8 {
        return false;
    }
    if lhs_metadata[2].shape != vec![rows, cols] || lhs_metadata[2].dtype != Dtype::I8 {
        return false;
    }
    if lhs_metadata[3].shape != vec![1] || lhs_metadata[3].dtype != Dtype::I8 {
        return false;
    }
    if lhs_metadata[4].shape != vec![rows, cols] || lhs_metadata[4].dtype != Dtype::I8 {
        return false;
    }
    if lhs_metadata[5].shape != vec![rows, cols] || lhs_metadata[5].dtype != Dtype::I8 {
        return false;
    }
    true
}


pub fn metadata_mvin_spad(
    egraph: &EGraph<TensorOp, TensorInfo>,
    lhs_eclasses: &Vec<Id>,
    _lhs_enodes: &Vec<Option<TensorOp>>,
) -> Vec<Option<String>> {
    assert_eq!(lhs_eclasses.len(), 6);
    let lhs_metadata: Vec<TensorInfo> = lhs_eclasses.iter().map(|id| egraph[*id].data.clone()).collect();
    let rows = lhs_metadata[1].shape[0];
    let cols = lhs_metadata[1].shape[1];
    let scale = {
        let ec = lhs_eclasses[3];
        let mut val = 0i32;
        for en in egraph[ec].nodes.iter() {
            if let TensorOp::OpConstant(v) = en {
                let num_str = v.split(':').next().unwrap_or(v);
                val = num_str.parse::<i32>().unwrap_or(0);
                break;
            }
        }
        val
    };

    let mut rhs_metadata = vec![None; 2];
    *rhs_metadata.last_mut().unwrap() = Some(format!("{},{},{}", rows, cols, scale).to_string());
    rhs_metadata
}


pub fn set_shapes_mvin_spad(
    egraph: &mut EGraph<TensorOp, TensorInfo>,
    lhs_eclasses: &Vec<Id>,
    rhs_eclasses: &Vec<Id>,
) {
    assert_eq!(lhs_eclasses.len(), 6);
    assert_eq!(rhs_eclasses.len(), 2);
    let lhs_metadata: Vec<TensorInfo> = lhs_eclasses.iter().map(|id| egraph[*id].data.clone()).collect();
    let rows = lhs_metadata[1].shape[0];
    let cols = lhs_metadata[1].shape[1];
    let scale = {
        let ec = lhs_eclasses[3];
        let mut val = 0i32;
        for en in egraph[ec].nodes.iter() {
            if let TensorOp::OpConstant(v) = en {
                let num_str = v.split(':').next().unwrap_or(v);
                val = num_str.parse::<i32>().unwrap_or(0);
                break;
            }
        }
        val
    };


    egraph.set_analysis_data(*rhs_eclasses.last().unwrap(),
        TensorInfo { shape: vec![rows, cols], dtype: Dtype::I8, is_const: false, });

}


pub fn precond_mvout_spad(egraph: &EGraph<TensorOp, TensorInfo>, lhs_eclasses: &Vec<Id>) -> bool {
    assert_eq!(lhs_eclasses.len(), 3);
    let lhs_metadata: Vec<TensorInfo> = lhs_eclasses.iter().map(|id| egraph[*id].data.clone()).collect();
    let shape_0 = &lhs_metadata[0].shape;
    if shape_0.len() <= 0 { return false; }
    let rows = shape_0[0];
    let shape_0 = &lhs_metadata[0].shape;
    if shape_0.len() <= 1 { return false; }
    let cols = shape_0[1];

    if lhs_metadata[0].shape != vec![rows, cols] || lhs_metadata[0].dtype != Dtype::I8 {
        return false;
    }
    if lhs_metadata[1].shape != vec![rows, cols] || lhs_metadata[1].dtype != Dtype::U8 {
        return false;
    }
    if lhs_metadata[2].shape != vec![rows*cols] || lhs_metadata[2].dtype != Dtype::U8 {
        return false;
    }
    true
}


pub fn metadata_mvout_spad(
    egraph: &EGraph<TensorOp, TensorInfo>,
    lhs_eclasses: &Vec<Id>,
    _lhs_enodes: &Vec<Option<TensorOp>>,
) -> Vec<Option<String>> {
    assert_eq!(lhs_eclasses.len(), 3);
    let lhs_metadata: Vec<TensorInfo> = lhs_eclasses.iter().map(|id| egraph[*id].data.clone()).collect();
    let rows = lhs_metadata[0].shape[0];
    let cols = lhs_metadata[0].shape[1];

    let mut rhs_metadata = vec![None; 2];
    *rhs_metadata.last_mut().unwrap() = Some(format!("{},{}", rows, cols).to_string());
    rhs_metadata
}


pub fn set_shapes_mvout_spad(
    egraph: &mut EGraph<TensorOp, TensorInfo>,
    lhs_eclasses: &Vec<Id>,
    rhs_eclasses: &Vec<Id>,
) {
    assert_eq!(lhs_eclasses.len(), 3);
    assert_eq!(rhs_eclasses.len(), 2);
    let lhs_metadata: Vec<TensorInfo> = lhs_eclasses.iter().map(|id| egraph[*id].data.clone()).collect();
    let rows = lhs_metadata[0].shape[0];
    let cols = lhs_metadata[0].shape[1];


    egraph.set_analysis_data(*rhs_eclasses.last().unwrap(),
        TensorInfo { shape: vec![rows*cols], dtype: Dtype::U8, is_const: false, });

}


pub fn precond_mvout_spad_relu(egraph: &EGraph<TensorOp, TensorInfo>, lhs_eclasses: &Vec<Id>) -> bool {
    assert_eq!(lhs_eclasses.len(), 6);
    let lhs_metadata: Vec<TensorInfo> = lhs_eclasses.iter().map(|id| egraph[*id].data.clone()).collect();
    let shape_0 = &lhs_metadata[0].shape;
    if shape_0.len() <= 0 { return false; }
    let rows = shape_0[0];
    let shape_0 = &lhs_metadata[0].shape;
    if shape_0.len() <= 1 { return false; }
    let cols = shape_0[1];

    if lhs_metadata[0].shape != vec![rows, cols] || lhs_metadata[0].dtype != Dtype::I8 {
        return false;
    }
    if lhs_metadata[1].shape != vec![1] || lhs_metadata[1].dtype != Dtype::I8 {
        return false;
    }
    if lhs_metadata[2].shape != vec![rows, cols] || lhs_metadata[2].dtype != Dtype::I8 {
        return false;
    }
    if lhs_metadata[3].shape != vec![rows, cols] || lhs_metadata[3].dtype != Dtype::I8 {
        return false;
    }
    if lhs_metadata[4].shape != vec![rows, cols] || lhs_metadata[4].dtype != Dtype::U8 {
        return false;
    }
    if lhs_metadata[5].shape != vec![rows*cols] || lhs_metadata[5].dtype != Dtype::U8 {
        return false;
    }
    true
}


pub fn metadata_mvout_spad_relu(
    egraph: &EGraph<TensorOp, TensorInfo>,
    lhs_eclasses: &Vec<Id>,
    _lhs_enodes: &Vec<Option<TensorOp>>,
) -> Vec<Option<String>> {
    assert_eq!(lhs_eclasses.len(), 6);
    let lhs_metadata: Vec<TensorInfo> = lhs_eclasses.iter().map(|id| egraph[*id].data.clone()).collect();
    let rows = lhs_metadata[0].shape[0];
    let cols = lhs_metadata[0].shape[1];

    let mut rhs_metadata = vec![None; 2];
    *rhs_metadata.last_mut().unwrap() = Some(format!("{},{}", rows, cols).to_string());
    rhs_metadata
}


pub fn set_shapes_mvout_spad_relu(
    egraph: &mut EGraph<TensorOp, TensorInfo>,
    lhs_eclasses: &Vec<Id>,
    rhs_eclasses: &Vec<Id>,
) {
    assert_eq!(lhs_eclasses.len(), 6);
    assert_eq!(rhs_eclasses.len(), 2);
    let lhs_metadata: Vec<TensorInfo> = lhs_eclasses.iter().map(|id| egraph[*id].data.clone()).collect();
    let rows = lhs_metadata[0].shape[0];
    let cols = lhs_metadata[0].shape[1];


    egraph.set_analysis_data(*rhs_eclasses.last().unwrap(),
        TensorInfo { shape: vec![rows*cols], dtype: Dtype::U8, is_const: false, });

}


pub fn precond_mvin_acc(egraph: &EGraph<TensorOp, TensorInfo>, lhs_eclasses: &Vec<Id>) -> bool {
    assert_eq!(lhs_eclasses.len(), 6);
    let lhs_metadata: Vec<TensorInfo> = lhs_eclasses.iter().map(|id| egraph[*id].data.clone()).collect();
    let shape_1 = &lhs_metadata[1].shape;
    if shape_1.len() <= 0 { return false; }
    let rows = shape_1[0];
    let shape_1 = &lhs_metadata[1].shape;
    if shape_1.len() <= 1 { return false; }
    let cols = shape_1[1];
    let scale = {
        let ec = lhs_eclasses[3];
        let mut val = 0i32;
        for en in egraph[ec].nodes.iter() {
            if let TensorOp::OpConstant(v) = en {
                let num_str = v.split(':').next().unwrap_or(v);
                val = num_str.parse::<i32>().unwrap_or(0);
                break;
            }
        }
        val
    };

    if lhs_metadata[0].shape != vec![rows*cols*4] || lhs_metadata[0].dtype != Dtype::U8 {
        return false;
    }
    if lhs_metadata[1].shape != vec![rows, cols, 4] || lhs_metadata[1].dtype != Dtype::U8 {
        return false;
    }
    if lhs_metadata[2].shape != vec![rows, cols] || lhs_metadata[2].dtype != Dtype::I32 {
        return false;
    }
    if lhs_metadata[3].shape != vec![1] || lhs_metadata[3].dtype != Dtype::I32 {
        return false;
    }
    if lhs_metadata[4].shape != vec![rows, cols] || lhs_metadata[4].dtype != Dtype::I32 {
        return false;
    }
    if lhs_metadata[5].shape != vec![rows, cols] || lhs_metadata[5].dtype != Dtype::I32 {
        return false;
    }
    true
}


pub fn metadata_mvin_acc(
    egraph: &EGraph<TensorOp, TensorInfo>,
    lhs_eclasses: &Vec<Id>,
    _lhs_enodes: &Vec<Option<TensorOp>>,
) -> Vec<Option<String>> {
    assert_eq!(lhs_eclasses.len(), 6);
    let lhs_metadata: Vec<TensorInfo> = lhs_eclasses.iter().map(|id| egraph[*id].data.clone()).collect();
    let rows = lhs_metadata[1].shape[0];
    let cols = lhs_metadata[1].shape[1];
    let scale = {
        let ec = lhs_eclasses[3];
        let mut val = 0i32;
        for en in egraph[ec].nodes.iter() {
            if let TensorOp::OpConstant(v) = en {
                let num_str = v.split(':').next().unwrap_or(v);
                val = num_str.parse::<i32>().unwrap_or(0);
                break;
            }
        }
        val
    };

    let mut rhs_metadata = vec![None; 2];
    *rhs_metadata.last_mut().unwrap() = Some(format!("{},{},{}", rows, cols, scale).to_string());
    rhs_metadata
}


pub fn set_shapes_mvin_acc(
    egraph: &mut EGraph<TensorOp, TensorInfo>,
    lhs_eclasses: &Vec<Id>,
    rhs_eclasses: &Vec<Id>,
) {
    assert_eq!(lhs_eclasses.len(), 6);
    assert_eq!(rhs_eclasses.len(), 2);
    let lhs_metadata: Vec<TensorInfo> = lhs_eclasses.iter().map(|id| egraph[*id].data.clone()).collect();
    let rows = lhs_metadata[1].shape[0];
    let cols = lhs_metadata[1].shape[1];
    let scale = {
        let ec = lhs_eclasses[3];
        let mut val = 0i32;
        for en in egraph[ec].nodes.iter() {
            if let TensorOp::OpConstant(v) = en {
                let num_str = v.split(':').next().unwrap_or(v);
                val = num_str.parse::<i32>().unwrap_or(0);
                break;
            }
        }
        val
    };


    egraph.set_analysis_data(*rhs_eclasses.last().unwrap(),
        TensorInfo { shape: vec![rows, cols], dtype: Dtype::I32, is_const: false, });

}


pub fn precond_mvout_acc(egraph: &EGraph<TensorOp, TensorInfo>, lhs_eclasses: &Vec<Id>) -> bool {
    assert_eq!(lhs_eclasses.len(), 3);
    let lhs_metadata: Vec<TensorInfo> = lhs_eclasses.iter().map(|id| egraph[*id].data.clone()).collect();
    let shape_0 = &lhs_metadata[0].shape;
    if shape_0.len() <= 0 { return false; }
    let rows = shape_0[0];
    let shape_0 = &lhs_metadata[0].shape;
    if shape_0.len() <= 1 { return false; }
    let cols = shape_0[1];

    if lhs_metadata[0].shape != vec![rows, cols] || lhs_metadata[0].dtype != Dtype::I32 {
        return false;
    }
    if lhs_metadata[1].shape != vec![rows, cols, 4] || lhs_metadata[1].dtype != Dtype::U8 {
        return false;
    }
    if lhs_metadata[2].shape != vec![rows*cols*4] || lhs_metadata[2].dtype != Dtype::U8 {
        return false;
    }
    true
}


pub fn metadata_mvout_acc(
    egraph: &EGraph<TensorOp, TensorInfo>,
    lhs_eclasses: &Vec<Id>,
    _lhs_enodes: &Vec<Option<TensorOp>>,
) -> Vec<Option<String>> {
    assert_eq!(lhs_eclasses.len(), 3);
    let lhs_metadata: Vec<TensorInfo> = lhs_eclasses.iter().map(|id| egraph[*id].data.clone()).collect();
    let rows = lhs_metadata[0].shape[0];
    let cols = lhs_metadata[0].shape[1];

    let mut rhs_metadata = vec![None; 2];
    *rhs_metadata.last_mut().unwrap() = Some(format!("{},{}", rows, cols).to_string());
    rhs_metadata
}


pub fn set_shapes_mvout_acc(
    egraph: &mut EGraph<TensorOp, TensorInfo>,
    lhs_eclasses: &Vec<Id>,
    rhs_eclasses: &Vec<Id>,
) {
    assert_eq!(lhs_eclasses.len(), 3);
    assert_eq!(rhs_eclasses.len(), 2);
    let lhs_metadata: Vec<TensorInfo> = lhs_eclasses.iter().map(|id| egraph[*id].data.clone()).collect();
    let rows = lhs_metadata[0].shape[0];
    let cols = lhs_metadata[0].shape[1];


    egraph.set_analysis_data(*rhs_eclasses.last().unwrap(),
        TensorInfo { shape: vec![rows*cols*4], dtype: Dtype::U8, is_const: false, });

}


pub fn precond_mvout_acc_relu(egraph: &EGraph<TensorOp, TensorInfo>, lhs_eclasses: &Vec<Id>) -> bool {
    assert_eq!(lhs_eclasses.len(), 6);
    let lhs_metadata: Vec<TensorInfo> = lhs_eclasses.iter().map(|id| egraph[*id].data.clone()).collect();
    let shape_0 = &lhs_metadata[0].shape;
    if shape_0.len() <= 0 { return false; }
    let rows = shape_0[0];
    let shape_0 = &lhs_metadata[0].shape;
    if shape_0.len() <= 1 { return false; }
    let cols = shape_0[1];

    if lhs_metadata[0].shape != vec![rows, cols] || lhs_metadata[0].dtype != Dtype::I32 {
        return false;
    }
    if lhs_metadata[1].shape != vec![1] || lhs_metadata[1].dtype != Dtype::I32 {
        return false;
    }
    if lhs_metadata[2].shape != vec![rows, cols] || lhs_metadata[2].dtype != Dtype::I32 {
        return false;
    }
    if lhs_metadata[3].shape != vec![rows, cols] || lhs_metadata[3].dtype != Dtype::I32 {
        return false;
    }
    if lhs_metadata[4].shape != vec![rows, cols, 4] || lhs_metadata[4].dtype != Dtype::U8 {
        return false;
    }
    if lhs_metadata[5].shape != vec![rows*cols*4] || lhs_metadata[5].dtype != Dtype::U8 {
        return false;
    }
    true
}


pub fn metadata_mvout_acc_relu(
    egraph: &EGraph<TensorOp, TensorInfo>,
    lhs_eclasses: &Vec<Id>,
    _lhs_enodes: &Vec<Option<TensorOp>>,
) -> Vec<Option<String>> {
    assert_eq!(lhs_eclasses.len(), 6);
    let lhs_metadata: Vec<TensorInfo> = lhs_eclasses.iter().map(|id| egraph[*id].data.clone()).collect();
    let rows = lhs_metadata[0].shape[0];
    let cols = lhs_metadata[0].shape[1];

    let mut rhs_metadata = vec![None; 2];
    *rhs_metadata.last_mut().unwrap() = Some(format!("{},{}", rows, cols).to_string());
    rhs_metadata
}


pub fn set_shapes_mvout_acc_relu(
    egraph: &mut EGraph<TensorOp, TensorInfo>,
    lhs_eclasses: &Vec<Id>,
    rhs_eclasses: &Vec<Id>,
) {
    assert_eq!(lhs_eclasses.len(), 6);
    assert_eq!(rhs_eclasses.len(), 2);
    let lhs_metadata: Vec<TensorInfo> = lhs_eclasses.iter().map(|id| egraph[*id].data.clone()).collect();
    let rows = lhs_metadata[0].shape[0];
    let cols = lhs_metadata[0].shape[1];


    egraph.set_analysis_data(*rhs_eclasses.last().unwrap(),
        TensorInfo { shape: vec![rows*cols*4], dtype: Dtype::U8, is_const: false, });

}


pub fn precond_mvin_acc_low(egraph: &EGraph<TensorOp, TensorInfo>, lhs_eclasses: &Vec<Id>) -> bool {
    assert_eq!(lhs_eclasses.len(), 7);
    let lhs_metadata: Vec<TensorInfo> = lhs_eclasses.iter().map(|id| egraph[*id].data.clone()).collect();
    let shape_1 = &lhs_metadata[1].shape;
    if shape_1.len() <= 0 { return false; }
    let rows = shape_1[0];
    let shape_1 = &lhs_metadata[1].shape;
    if shape_1.len() <= 1 { return false; }
    let cols = shape_1[1];
    let scale = {
        let ec = lhs_eclasses[3];
        let mut val = 0i32;
        for en in egraph[ec].nodes.iter() {
            if let TensorOp::OpConstant(v) = en {
                let num_str = v.split(':').next().unwrap_or(v);
                val = num_str.parse::<i32>().unwrap_or(0);
                break;
            }
        }
        val
    };

    if lhs_metadata[0].shape != vec![rows*cols] || lhs_metadata[0].dtype != Dtype::U8 {
        return false;
    }
    if lhs_metadata[1].shape != vec![rows, cols] || lhs_metadata[1].dtype != Dtype::U8 {
        return false;
    }
    if lhs_metadata[2].shape != vec![rows, cols] || lhs_metadata[2].dtype != Dtype::I8 {
        return false;
    }
    if lhs_metadata[3].shape != vec![1] || lhs_metadata[3].dtype != Dtype::I8 {
        return false;
    }
    if lhs_metadata[4].shape != vec![rows, cols] || lhs_metadata[4].dtype != Dtype::I8 {
        return false;
    }
    if lhs_metadata[5].shape != vec![rows, cols] || lhs_metadata[5].dtype != Dtype::I8 {
        return false;
    }
    if lhs_metadata[6].shape != vec![rows, cols] || lhs_metadata[6].dtype != Dtype::I32 {
        return false;
    }
    true
}


pub fn metadata_mvin_acc_low(
    egraph: &EGraph<TensorOp, TensorInfo>,
    lhs_eclasses: &Vec<Id>,
    _lhs_enodes: &Vec<Option<TensorOp>>,
) -> Vec<Option<String>> {
    assert_eq!(lhs_eclasses.len(), 7);
    let lhs_metadata: Vec<TensorInfo> = lhs_eclasses.iter().map(|id| egraph[*id].data.clone()).collect();
    let rows = lhs_metadata[1].shape[0];
    let cols = lhs_metadata[1].shape[1];
    let scale = {
        let ec = lhs_eclasses[3];
        let mut val = 0i32;
        for en in egraph[ec].nodes.iter() {
            if let TensorOp::OpConstant(v) = en {
                let num_str = v.split(':').next().unwrap_or(v);
                val = num_str.parse::<i32>().unwrap_or(0);
                break;
            }
        }
        val
    };

    let mut rhs_metadata = vec![None; 2];
    *rhs_metadata.last_mut().unwrap() = Some(format!("{},{},{}", rows, cols, scale).to_string());
    rhs_metadata
}


pub fn set_shapes_mvin_acc_low(
    egraph: &mut EGraph<TensorOp, TensorInfo>,
    lhs_eclasses: &Vec<Id>,
    rhs_eclasses: &Vec<Id>,
) {
    assert_eq!(lhs_eclasses.len(), 7);
    assert_eq!(rhs_eclasses.len(), 2);
    let lhs_metadata: Vec<TensorInfo> = lhs_eclasses.iter().map(|id| egraph[*id].data.clone()).collect();
    let rows = lhs_metadata[1].shape[0];
    let cols = lhs_metadata[1].shape[1];
    let scale = {
        let ec = lhs_eclasses[3];
        let mut val = 0i32;
        for en in egraph[ec].nodes.iter() {
            if let TensorOp::OpConstant(v) = en {
                let num_str = v.split(':').next().unwrap_or(v);
                val = num_str.parse::<i32>().unwrap_or(0);
                break;
            }
        }
        val
    };


    egraph.set_analysis_data(*rhs_eclasses.last().unwrap(),
        TensorInfo { shape: vec![rows, cols], dtype: Dtype::I32, is_const: false, });

}


pub fn precond_mvin_acc_low_add(egraph: &EGraph<TensorOp, TensorInfo>, lhs_eclasses: &Vec<Id>) -> bool {
    assert_eq!(lhs_eclasses.len(), 9);
    let lhs_metadata: Vec<TensorInfo> = lhs_eclasses.iter().map(|id| egraph[*id].data.clone()).collect();
    let shape_0 = &lhs_metadata[0].shape;
    if shape_0.len() <= 0 { return false; }
    let rows = shape_0[0];
    let shape_0 = &lhs_metadata[0].shape;
    if shape_0.len() <= 1 { return false; }
    let cols = shape_0[1];
    let scale = {
        let ec = lhs_eclasses[4];
        let mut val = 0i32;
        for en in egraph[ec].nodes.iter() {
            if let TensorOp::OpConstant(v) = en {
                let num_str = v.split(':').next().unwrap_or(v);
                val = num_str.parse::<i32>().unwrap_or(0);
                break;
            }
        }
        val
    };

    if lhs_metadata[0].shape != vec![rows, cols] || lhs_metadata[0].dtype != Dtype::I32 {
        return false;
    }
    if lhs_metadata[1].shape != vec![rows*cols] || lhs_metadata[1].dtype != Dtype::U8 {
        return false;
    }
    if lhs_metadata[2].shape != vec![rows, cols] || lhs_metadata[2].dtype != Dtype::U8 {
        return false;
    }
    if lhs_metadata[3].shape != vec![rows, cols] || lhs_metadata[3].dtype != Dtype::I8 {
        return false;
    }
    if lhs_metadata[4].shape != vec![1] || lhs_metadata[4].dtype != Dtype::I8 {
        return false;
    }
    if lhs_metadata[5].shape != vec![rows, cols] || lhs_metadata[5].dtype != Dtype::I8 {
        return false;
    }
    if lhs_metadata[6].shape != vec![rows, cols] || lhs_metadata[6].dtype != Dtype::I8 {
        return false;
    }
    if lhs_metadata[7].shape != vec![rows, cols] || lhs_metadata[7].dtype != Dtype::I32 {
        return false;
    }
    if lhs_metadata[8].shape != vec![rows, cols] || lhs_metadata[8].dtype != Dtype::I32 {
        return false;
    }
    true
}


pub fn metadata_mvin_acc_low_add(
    egraph: &EGraph<TensorOp, TensorInfo>,
    lhs_eclasses: &Vec<Id>,
    _lhs_enodes: &Vec<Option<TensorOp>>,
) -> Vec<Option<String>> {
    assert_eq!(lhs_eclasses.len(), 9);
    let lhs_metadata: Vec<TensorInfo> = lhs_eclasses.iter().map(|id| egraph[*id].data.clone()).collect();
    let rows = lhs_metadata[0].shape[0];
    let cols = lhs_metadata[0].shape[1];
    let scale = {
        let ec = lhs_eclasses[4];
        let mut val = 0i32;
        for en in egraph[ec].nodes.iter() {
            if let TensorOp::OpConstant(v) = en {
                let num_str = v.split(':').next().unwrap_or(v);
                val = num_str.parse::<i32>().unwrap_or(0);
                break;
            }
        }
        val
    };

    let mut rhs_metadata = vec![None; 3];
    *rhs_metadata.last_mut().unwrap() = Some(format!("{},{},{}", rows, cols, scale).to_string());
    rhs_metadata
}


pub fn set_shapes_mvin_acc_low_add(
    egraph: &mut EGraph<TensorOp, TensorInfo>,
    lhs_eclasses: &Vec<Id>,
    rhs_eclasses: &Vec<Id>,
) {
    assert_eq!(lhs_eclasses.len(), 9);
    assert_eq!(rhs_eclasses.len(), 3);
    let lhs_metadata: Vec<TensorInfo> = lhs_eclasses.iter().map(|id| egraph[*id].data.clone()).collect();
    let rows = lhs_metadata[0].shape[0];
    let cols = lhs_metadata[0].shape[1];
    let scale = {
        let ec = lhs_eclasses[4];
        let mut val = 0i32;
        for en in egraph[ec].nodes.iter() {
            if let TensorOp::OpConstant(v) = en {
                let num_str = v.split(':').next().unwrap_or(v);
                val = num_str.parse::<i32>().unwrap_or(0);
                break;
            }
        }
        val
    };


    egraph.set_analysis_data(*rhs_eclasses.last().unwrap(),
        TensorInfo { shape: vec![rows, cols], dtype: Dtype::I32, is_const: false, });

}


pub fn precond_mvout_acc_low(egraph: &EGraph<TensorOp, TensorInfo>, lhs_eclasses: &Vec<Id>) -> bool {
    assert_eq!(lhs_eclasses.len(), 4);
    let lhs_metadata: Vec<TensorInfo> = lhs_eclasses.iter().map(|id| egraph[*id].data.clone()).collect();
    let shape_0 = &lhs_metadata[0].shape;
    if shape_0.len() <= 0 { return false; }
    let rows = shape_0[0];
    let shape_0 = &lhs_metadata[0].shape;
    if shape_0.len() <= 1 { return false; }
    let cols = shape_0[1];

    if lhs_metadata[0].shape != vec![rows, cols] || lhs_metadata[0].dtype != Dtype::I32 {
        return false;
    }
    if lhs_metadata[1].shape != vec![rows, cols] || lhs_metadata[1].dtype != Dtype::I8 {
        return false;
    }
    if lhs_metadata[2].shape != vec![rows, cols] || lhs_metadata[2].dtype != Dtype::U8 {
        return false;
    }
    if lhs_metadata[3].shape != vec![rows*cols] || lhs_metadata[3].dtype != Dtype::U8 {
        return false;
    }
    true
}


pub fn metadata_mvout_acc_low(
    egraph: &EGraph<TensorOp, TensorInfo>,
    lhs_eclasses: &Vec<Id>,
    _lhs_enodes: &Vec<Option<TensorOp>>,
) -> Vec<Option<String>> {
    assert_eq!(lhs_eclasses.len(), 4);
    let lhs_metadata: Vec<TensorInfo> = lhs_eclasses.iter().map(|id| egraph[*id].data.clone()).collect();
    let rows = lhs_metadata[0].shape[0];
    let cols = lhs_metadata[0].shape[1];

    let mut rhs_metadata = vec![None; 2];
    *rhs_metadata.last_mut().unwrap() = Some(format!("{},{}", rows, cols).to_string());
    rhs_metadata
}


pub fn set_shapes_mvout_acc_low(
    egraph: &mut EGraph<TensorOp, TensorInfo>,
    lhs_eclasses: &Vec<Id>,
    rhs_eclasses: &Vec<Id>,
) {
    assert_eq!(lhs_eclasses.len(), 4);
    assert_eq!(rhs_eclasses.len(), 2);
    let lhs_metadata: Vec<TensorInfo> = lhs_eclasses.iter().map(|id| egraph[*id].data.clone()).collect();
    let rows = lhs_metadata[0].shape[0];
    let cols = lhs_metadata[0].shape[1];


    egraph.set_analysis_data(*rhs_eclasses.last().unwrap(),
        TensorInfo { shape: vec![rows*cols], dtype: Dtype::U8, is_const: false, });

}


pub fn precond_mvout_acc_low_relu(egraph: &EGraph<TensorOp, TensorInfo>, lhs_eclasses: &Vec<Id>) -> bool {
    assert_eq!(lhs_eclasses.len(), 7);
    let lhs_metadata: Vec<TensorInfo> = lhs_eclasses.iter().map(|id| egraph[*id].data.clone()).collect();
    let shape_0 = &lhs_metadata[0].shape;
    if shape_0.len() <= 0 { return false; }
    let rows = shape_0[0];
    let shape_0 = &lhs_metadata[0].shape;
    if shape_0.len() <= 1 { return false; }
    let cols = shape_0[1];

    if lhs_metadata[0].shape != vec![rows, cols] || lhs_metadata[0].dtype != Dtype::I32 {
        return false;
    }
    if lhs_metadata[1].shape != vec![1] || lhs_metadata[1].dtype != Dtype::I32 {
        return false;
    }
    if lhs_metadata[2].shape != vec![rows, cols] || lhs_metadata[2].dtype != Dtype::I32 {
        return false;
    }
    if lhs_metadata[3].shape != vec![rows, cols] || lhs_metadata[3].dtype != Dtype::I32 {
        return false;
    }
    if lhs_metadata[4].shape != vec![rows, cols] || lhs_metadata[4].dtype != Dtype::I8 {
        return false;
    }
    if lhs_metadata[5].shape != vec![rows, cols] || lhs_metadata[5].dtype != Dtype::U8 {
        return false;
    }
    if lhs_metadata[6].shape != vec![rows*cols] || lhs_metadata[6].dtype != Dtype::U8 {
        return false;
    }
    true
}


pub fn metadata_mvout_acc_low_relu(
    egraph: &EGraph<TensorOp, TensorInfo>,
    lhs_eclasses: &Vec<Id>,
    _lhs_enodes: &Vec<Option<TensorOp>>,
) -> Vec<Option<String>> {
    assert_eq!(lhs_eclasses.len(), 7);
    let lhs_metadata: Vec<TensorInfo> = lhs_eclasses.iter().map(|id| egraph[*id].data.clone()).collect();
    let rows = lhs_metadata[0].shape[0];
    let cols = lhs_metadata[0].shape[1];

    let mut rhs_metadata = vec![None; 2];
    *rhs_metadata.last_mut().unwrap() = Some(format!("{},{}", rows, cols).to_string());
    rhs_metadata
}


pub fn set_shapes_mvout_acc_low_relu(
    egraph: &mut EGraph<TensorOp, TensorInfo>,
    lhs_eclasses: &Vec<Id>,
    rhs_eclasses: &Vec<Id>,
) {
    assert_eq!(lhs_eclasses.len(), 7);
    assert_eq!(rhs_eclasses.len(), 2);
    let lhs_metadata: Vec<TensorInfo> = lhs_eclasses.iter().map(|id| egraph[*id].data.clone()).collect();
    let rows = lhs_metadata[0].shape[0];
    let cols = lhs_metadata[0].shape[1];


    egraph.set_analysis_data(*rhs_eclasses.last().unwrap(),
        TensorInfo { shape: vec![rows*cols], dtype: Dtype::U8, is_const: false, });

}


pub fn precond_matmul8(egraph: &EGraph<TensorOp, TensorInfo>, lhs_eclasses: &Vec<Id>) -> bool {
    assert_eq!(lhs_eclasses.len(), 3);
    let lhs_metadata: Vec<TensorInfo> = lhs_eclasses.iter().map(|id| egraph[*id].data.clone()).collect();
    let shape_0 = &lhs_metadata[0].shape;
    if shape_0.len() <= 0 { return false; }
    let DIM_I = shape_0[0];
    let shape_1 = &lhs_metadata[1].shape;
    if shape_1.len() <= 1 { return false; }
    let DIM_J = shape_1[1];
    let shape_0 = &lhs_metadata[0].shape;
    if shape_0.len() <= 1 { return false; }
    let DIM_K = shape_0[1];

    if lhs_metadata[0].shape != vec![DIM_I, DIM_K] || lhs_metadata[0].dtype != Dtype::I8 {
        return false;
    }
    if lhs_metadata[1].shape != vec![DIM_K, DIM_J] || lhs_metadata[1].dtype != Dtype::I8 {
        return false;
    }
    if lhs_metadata[2].shape != vec![DIM_I, DIM_J] || lhs_metadata[2].dtype != Dtype::I8 {
        return false;
    }
    true
}


pub fn metadata_matmul8(
    egraph: &EGraph<TensorOp, TensorInfo>,
    lhs_eclasses: &Vec<Id>,
    _lhs_enodes: &Vec<Option<TensorOp>>,
) -> Vec<Option<String>> {
    assert_eq!(lhs_eclasses.len(), 3);
    let lhs_metadata: Vec<TensorInfo> = lhs_eclasses.iter().map(|id| egraph[*id].data.clone()).collect();
    let DIM_I = lhs_metadata[0].shape[0];
    let DIM_J = lhs_metadata[1].shape[1];
    let DIM_K = lhs_metadata[0].shape[1];

    let mut rhs_metadata = vec![None; 3];
    *rhs_metadata.last_mut().unwrap() = Some(format!("{},{},{}", DIM_I, DIM_J, DIM_K).to_string());
    rhs_metadata
}


pub fn set_shapes_matmul8(
    egraph: &mut EGraph<TensorOp, TensorInfo>,
    lhs_eclasses: &Vec<Id>,
    rhs_eclasses: &Vec<Id>,
) {
    assert_eq!(lhs_eclasses.len(), 3);
    assert_eq!(rhs_eclasses.len(), 3);
    let lhs_metadata: Vec<TensorInfo> = lhs_eclasses.iter().map(|id| egraph[*id].data.clone()).collect();
    let DIM_I = lhs_metadata[0].shape[0];
    let DIM_J = lhs_metadata[1].shape[1];
    let DIM_K = lhs_metadata[0].shape[1];


    egraph.set_analysis_data(*rhs_eclasses.last().unwrap(),
        TensorInfo { shape: vec![DIM_I, DIM_J], dtype: Dtype::I8, is_const: false, });

}


pub fn precond_matmul32(egraph: &EGraph<TensorOp, TensorInfo>, lhs_eclasses: &Vec<Id>) -> bool {
    assert_eq!(lhs_eclasses.len(), 5);
    let lhs_metadata: Vec<TensorInfo> = lhs_eclasses.iter().map(|id| egraph[*id].data.clone()).collect();
    let shape_0 = &lhs_metadata[0].shape;
    if shape_0.len() <= 0 { return false; }
    let DIM_I = shape_0[0];
    let shape_1 = &lhs_metadata[1].shape;
    if shape_1.len() <= 1 { return false; }
    let DIM_J = shape_1[1];
    let shape_0 = &lhs_metadata[0].shape;
    if shape_0.len() <= 1 { return false; }
    let DIM_K = shape_0[1];

    if lhs_metadata[0].shape != vec![DIM_I, DIM_K] || lhs_metadata[0].dtype != Dtype::I8 {
        return false;
    }
    if lhs_metadata[1].shape != vec![DIM_K, DIM_J] || lhs_metadata[1].dtype != Dtype::I8 {
        return false;
    }
    if lhs_metadata[2].shape != vec![DIM_I, DIM_K] || lhs_metadata[2].dtype != Dtype::I32 {
        return false;
    }
    if lhs_metadata[3].shape != vec![DIM_K, DIM_J] || lhs_metadata[3].dtype != Dtype::I32 {
        return false;
    }
    if lhs_metadata[4].shape != vec![DIM_I, DIM_J] || lhs_metadata[4].dtype != Dtype::I32 {
        return false;
    }
    true
}


pub fn metadata_matmul32(
    egraph: &EGraph<TensorOp, TensorInfo>,
    lhs_eclasses: &Vec<Id>,
    _lhs_enodes: &Vec<Option<TensorOp>>,
) -> Vec<Option<String>> {
    assert_eq!(lhs_eclasses.len(), 5);
    let lhs_metadata: Vec<TensorInfo> = lhs_eclasses.iter().map(|id| egraph[*id].data.clone()).collect();
    let DIM_I = lhs_metadata[0].shape[0];
    let DIM_J = lhs_metadata[1].shape[1];
    let DIM_K = lhs_metadata[0].shape[1];

    let mut rhs_metadata = vec![None; 3];
    *rhs_metadata.last_mut().unwrap() = Some(format!("{},{},{}", DIM_I, DIM_J, DIM_K).to_string());
    rhs_metadata
}


pub fn set_shapes_matmul32(
    egraph: &mut EGraph<TensorOp, TensorInfo>,
    lhs_eclasses: &Vec<Id>,
    rhs_eclasses: &Vec<Id>,
) {
    assert_eq!(lhs_eclasses.len(), 5);
    assert_eq!(rhs_eclasses.len(), 3);
    let lhs_metadata: Vec<TensorInfo> = lhs_eclasses.iter().map(|id| egraph[*id].data.clone()).collect();
    let DIM_I = lhs_metadata[0].shape[0];
    let DIM_J = lhs_metadata[1].shape[1];
    let DIM_K = lhs_metadata[0].shape[1];


    egraph.set_analysis_data(*rhs_eclasses.last().unwrap(),
        TensorInfo { shape: vec![DIM_I, DIM_J], dtype: Dtype::I32, is_const: false, });

}


pub fn precond_mac8(egraph: &EGraph<TensorOp, TensorInfo>, lhs_eclasses: &Vec<Id>) -> bool {
    assert_eq!(lhs_eclasses.len(), 5);
    let lhs_metadata: Vec<TensorInfo> = lhs_eclasses.iter().map(|id| egraph[*id].data.clone()).collect();
    let shape_0 = &lhs_metadata[0].shape;
    if shape_0.len() <= 0 { return false; }
    let DIM_I = shape_0[0];
    let shape_1 = &lhs_metadata[1].shape;
    if shape_1.len() <= 1 { return false; }
    let DIM_J = shape_1[1];
    let shape_0 = &lhs_metadata[0].shape;
    if shape_0.len() <= 1 { return false; }
    let DIM_K = shape_0[1];

    if lhs_metadata[0].shape != vec![DIM_I, DIM_K] || lhs_metadata[0].dtype != Dtype::I8 {
        return false;
    }
    if lhs_metadata[1].shape != vec![DIM_K, DIM_J] || lhs_metadata[1].dtype != Dtype::I8 {
        return false;
    }
    if lhs_metadata[2].shape != vec![DIM_I, DIM_J] || lhs_metadata[2].dtype != Dtype::I8 {
        return false;
    }
    if lhs_metadata[3].shape != vec![DIM_I, DIM_J] || lhs_metadata[3].dtype != Dtype::I8 {
        return false;
    }
    if lhs_metadata[4].shape != vec![DIM_I, DIM_J] || lhs_metadata[4].dtype != Dtype::I8 {
        return false;
    }
    true
}


pub fn metadata_mac8(
    egraph: &EGraph<TensorOp, TensorInfo>,
    lhs_eclasses: &Vec<Id>,
    _lhs_enodes: &Vec<Option<TensorOp>>,
) -> Vec<Option<String>> {
    assert_eq!(lhs_eclasses.len(), 5);
    let lhs_metadata: Vec<TensorInfo> = lhs_eclasses.iter().map(|id| egraph[*id].data.clone()).collect();
    let DIM_I = lhs_metadata[0].shape[0];
    let DIM_J = lhs_metadata[1].shape[1];
    let DIM_K = lhs_metadata[0].shape[1];

    let mut rhs_metadata = vec![None; 4];
    *rhs_metadata.last_mut().unwrap() = Some(format!("{},{},{}", DIM_I, DIM_J, DIM_K).to_string());
    rhs_metadata
}


pub fn set_shapes_mac8(
    egraph: &mut EGraph<TensorOp, TensorInfo>,
    lhs_eclasses: &Vec<Id>,
    rhs_eclasses: &Vec<Id>,
) {
    assert_eq!(lhs_eclasses.len(), 5);
    assert_eq!(rhs_eclasses.len(), 4);
    let lhs_metadata: Vec<TensorInfo> = lhs_eclasses.iter().map(|id| egraph[*id].data.clone()).collect();
    let DIM_I = lhs_metadata[0].shape[0];
    let DIM_J = lhs_metadata[1].shape[1];
    let DIM_K = lhs_metadata[0].shape[1];


    egraph.set_analysis_data(*rhs_eclasses.last().unwrap(),
        TensorInfo { shape: vec![DIM_I, DIM_J], dtype: Dtype::I8, is_const: false, });

}


pub fn precond_mac32(egraph: &EGraph<TensorOp, TensorInfo>, lhs_eclasses: &Vec<Id>) -> bool {
    assert_eq!(lhs_eclasses.len(), 7);
    let lhs_metadata: Vec<TensorInfo> = lhs_eclasses.iter().map(|id| egraph[*id].data.clone()).collect();
    let shape_0 = &lhs_metadata[0].shape;
    if shape_0.len() <= 0 { return false; }
    let DIM_I = shape_0[0];
    let shape_1 = &lhs_metadata[1].shape;
    if shape_1.len() <= 1 { return false; }
    let DIM_J = shape_1[1];
    let shape_0 = &lhs_metadata[0].shape;
    if shape_0.len() <= 1 { return false; }
    let DIM_K = shape_0[1];

    if lhs_metadata[0].shape != vec![DIM_I, DIM_K] || lhs_metadata[0].dtype != Dtype::I8 {
        return false;
    }
    if lhs_metadata[1].shape != vec![DIM_K, DIM_J] || lhs_metadata[1].dtype != Dtype::I8 {
        return false;
    }
    if lhs_metadata[2].shape != vec![DIM_I, DIM_J] || lhs_metadata[2].dtype != Dtype::I32 {
        return false;
    }
    if lhs_metadata[3].shape != vec![DIM_I, DIM_K] || lhs_metadata[3].dtype != Dtype::I32 {
        return false;
    }
    if lhs_metadata[4].shape != vec![DIM_K, DIM_J] || lhs_metadata[4].dtype != Dtype::I32 {
        return false;
    }
    if lhs_metadata[5].shape != vec![DIM_I, DIM_J] || lhs_metadata[5].dtype != Dtype::I32 {
        return false;
    }
    if lhs_metadata[6].shape != vec![DIM_I, DIM_J] || lhs_metadata[6].dtype != Dtype::I32 {
        return false;
    }
    true
}


pub fn metadata_mac32(
    egraph: &EGraph<TensorOp, TensorInfo>,
    lhs_eclasses: &Vec<Id>,
    _lhs_enodes: &Vec<Option<TensorOp>>,
) -> Vec<Option<String>> {
    assert_eq!(lhs_eclasses.len(), 7);
    let lhs_metadata: Vec<TensorInfo> = lhs_eclasses.iter().map(|id| egraph[*id].data.clone()).collect();
    let DIM_I = lhs_metadata[0].shape[0];
    let DIM_J = lhs_metadata[1].shape[1];
    let DIM_K = lhs_metadata[0].shape[1];

    let mut rhs_metadata = vec![None; 4];
    *rhs_metadata.last_mut().unwrap() = Some(format!("{},{},{}", DIM_I, DIM_J, DIM_K).to_string());
    rhs_metadata
}


pub fn set_shapes_mac32(
    egraph: &mut EGraph<TensorOp, TensorInfo>,
    lhs_eclasses: &Vec<Id>,
    rhs_eclasses: &Vec<Id>,
) {
    assert_eq!(lhs_eclasses.len(), 7);
    assert_eq!(rhs_eclasses.len(), 4);
    let lhs_metadata: Vec<TensorInfo> = lhs_eclasses.iter().map(|id| egraph[*id].data.clone()).collect();
    let DIM_I = lhs_metadata[0].shape[0];
    let DIM_J = lhs_metadata[1].shape[1];
    let DIM_K = lhs_metadata[0].shape[1];


    egraph.set_analysis_data(*rhs_eclasses.last().unwrap(),
        TensorInfo { shape: vec![DIM_I, DIM_J], dtype: Dtype::I32, is_const: false, });

}



