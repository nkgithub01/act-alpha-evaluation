use egg::{EGraph, Id};

use crate::ir::egraph::{TensorInfo, TensorOp};

pub fn enforce_alpha_injectivity(egraph: &mut EGraph<TensorOp, TensorInfo>) -> bool {
    let mut unions: Vec<(Id, Id)> = vec![];

    for class in egraph.classes() {
        let mut d0_children: Vec<Id> = vec![];
        let mut d1_children: Vec<Id> = vec![];
        let mut d3_children: Vec<Id> = vec![];
        let mut d2_children: Vec<Id> = vec![];

        for node in &class.nodes {
            match node {
                TensorOp::AlphaHBM(child) => d0_children.push(egraph.find(*child)),
                TensorOp::AlphaD1(child) => d1_children.push(egraph.find(*child)),
                TensorOp::AlphaD3(child) => d3_children.push(egraph.find(*child)),
                TensorOp::AlphaD2(child) => d2_children.push(egraph.find(*child)),
                _ => {}
            }
        }

        add_child_unions(&mut unions, &d0_children);
        add_child_unions(&mut unions, &d1_children);
        add_child_unions(&mut unions, &d3_children);
        add_child_unions(&mut unions, &d2_children);
    }

    let mut changed = false;
    for (a, b) in unions {
        changed |= egraph.union(a, b);
    }

    if changed {
        egraph.rebuild();
    }

    changed
}

fn add_child_unions(unions: &mut Vec<(Id, Id)>, children: &[Id]) {
    let Some((&first, rest)) = children.split_first() else {
        return;
    };

    for &child in rest {
        if child != first {
            unions.push((first, child));
        }
    }
}
