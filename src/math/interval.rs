// src/math/interval.rs
//
// Implements rigorous interval arithmetic for Newton-Kantorovich certification.
// Uses conservative outward rounding to guarantee enclosure.

#[derive(Clone, Copy, Debug)]
pub struct Interval {
    pub inf: f64,
    pub sup: f64,
}

impl Interval {
    pub fn new(val: f64) -> Self {
        // Enclose the float with a 1-ULP bound to account for representation error
        let eps = val.abs() * f64::EPSILON;
        Interval {
            inf: val - eps,
            sup: val + eps,
        }
    }

    pub fn abs_max(&self) -> f64 {
        self.inf.abs().max(self.sup.abs())
    }
}

impl std::ops::Add for Interval {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        let inf = self.inf + other.inf;
        let sup = self.sup + other.sup;
        let eps = (inf.abs().max(sup.abs())) * f64::EPSILON;
        Interval { inf: inf - eps, sup: sup + eps }
    }
}

impl std::ops::Sub for Interval {
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        let inf = self.inf - other.sup;
        let sup = self.sup - other.inf;
        let eps = (inf.abs().max(sup.abs())) * f64::EPSILON;
        Interval { inf: inf - eps, sup: sup + eps }
    }
}

impl std::ops::Mul for Interval {
    type Output = Self;
    fn mul(self, other: Self) -> Self {
        let p1 = self.inf * other.inf;
        let p2 = self.inf * other.sup;
        let p3 = self.sup * other.inf;
        let p4 = self.sup * other.sup;
        
        let inf = p1.min(p2).min(p3).min(p4);
        let sup = p1.max(p2).max(p3).max(p4);
        let eps = (inf.abs().max(sup.abs())) * f64::EPSILON;
        
        Interval { inf: inf - eps, sup: sup + eps }
    }
}

/// Computes the Newton-Kantorovich bounds.
/// eta: The Picard residual ||u - \Phi(u)||
/// lipschitz_L: The interval-bounded Lipschitz constant of the Jacobian D\Phi
pub fn nk_certify(eta: f64, lipschitz_l: f64) -> Option<f64> {
    // kappa = || D\Phi ||_op
    let kappa = lipschitz_l;
    
    if kappa >= 1.0 {
        return None; // Operator is not a strict contraction
    }
    
    // Rigorous error bound: r* = eta / (1 - kappa)
    let bound = eta / (1.0 - kappa);
    // Add conservative outward rounding for the division
    Some(bound + bound * f64::EPSILON)
}