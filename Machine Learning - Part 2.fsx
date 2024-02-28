(**
# Machine Learning - Part 2

## Finding models from Data

In this article I'm exploring the use of regression techniques to find models directly from data and find analytic model solutions using the Sparse Identification of Non-Linear Dynamics or SINdy.

There is a lot of literature online and talks on this subject so here I aim to give my own interpretation and explore concrete application of the approaches in code.

## Introduction

The goal of model discovery is to find the governing system or equations that describe the time evolution of the system. This can be written as:

$$
\frac { \delta f } { \delta t } = f( \bar x,t)
$$

Also suppose we have collected time measurements of the state of our system such that:

$$
\text {given} \quad \bar y(t_k) \quad k = 1, 2, ... K
$$

We can compute the derivatives from the measurement data directly although noting this approach is sensitive to any noise in the data. Other techniques exist to compute the derivatives that are more robust to noise.

$$
\dot y (k) =  \frac {y_k - y_{k-1}} {\Delta t} = b
$$

In the example given in this article we compute the derivatives from the model directly to get the best possible data to use in the modelling problem. In a real data driven problem where the model is unknown
this is not possible so the derivatives will need to be carefully crafted separately.

Lets propose our system will contain a small number of terms in the final model, that is the model is sparse. We want to find the set of terms which best describe the system from a larger set of possible candidate or library of possible functions.

This possible candidate set of measurement functions could then be written as:

$$
A = \bigg \lbrack 
\bar x
\quad
\bar y
\quad
\bar z
\quad
\bar x^2
\quad
\bar x \bar y
\quad
\bar x \bar z
\quad
\bar y^2
\quad
\bar y \bar z
\quad
\bar z^2
\quad
sin(\bar y)
\bigg \rbrack
$$
We can now pose the problem as the sparse solution to the form:

$$
A v = b
$$

If we perform a sparsity promoting regression for $v$, we have the weights of each of the terms that are active in the library which make up the systems dynamics.
Algorithms which can be used to perform a sparse regression such as Basis Pursuit, LASSO (Least Absolute Shrinkage and Selection Operator), or STLS (Sequential Thresholded Least Squares).

To demonstrate the methods we construct an example model using the classic lorentz system otherwise known as the butterfly attractor. The equations for this system are given as:

$$
\displaylines{
\dot x = \sigma (y - x)   \\
\dot y = x (\rho - z) - y \\
\dot z = xy - \beta z     \\
}
$$

By setting $
\rho = 28, \: \sigma = 10, \: \beta = 8/3
$ we get the following system dynamics plotted in 3D.
*)

#r "nuget: MathNet.Numerics, 5.0.0"
#r "nuget: MathNet.Numerics.FSharp, 5.0.0"
#r "nuget: Plotly.NET, 4.2.0"
#r "nuget: Plotly.NET.Interactive, 4.2.0"
(*** hide ***)

open Plotly.NET
open Plotly.NET.LayoutObjects
open MathNet.Numerics.OdeSolvers
open MathNet.Numerics.LinearAlgebra
open MathNet.Numerics.LinearAlgebra.Double

let xAxis =
    let tmp = LinearAxis()
    tmp?title <- "xAxis"
    tmp?showgrid <- true
    tmp?showline <- true
    tmp

let yAxis =
    let tmp = LinearAxis()
    tmp?title <- "yAxis"
    tmp?showgrid <- true
    tmp?showline <- true
    tmp

let margin =
    let tmp = Margin()
    tmp?t <- 0.0
    tmp?l <- 0.0
    tmp?b <- 0.0
    tmp

let scene =
    let eye = CameraEye()
    eye?x <- 1.88
    eye?y <- -2.12
    eye?z <- 0.96
    let cam = Camera()
    cam?eye <- eye
    let tmp = Scene()
    tmp?camera <- cam
    tmp

let layout =
    let tmp = Layout()
    tmp?xaxis <- xAxis
    tmp?yaxis <- yAxis
    tmp?showlegend <- true
    tmp?width <- 1200.0
    tmp?height <- 800.0
    tmp?margin <- margin
    tmp?scene <- scene
    tmp

let line =
    let tmp = Line()
    tmp?width <- 3.0
    tmp

let trace name line (soln: float Matrix) =
    let tmp = Trace("scatter3d")
    tmp?x <- (soln[*, 0])
    tmp?y <- (soln[*, 1])
    tmp?z <- (soln[*, 2])
    tmp?line <- line
    tmp?mode <- "lines"
    tmp?name <- name
    tmp

(** *)

open MathNet.Numerics.OdeSolvers
open MathNet.Numerics.LinearAlgebra
open MathNet.Numerics.LinearAlgebra.Double

// The canonical lorentz system equations
let lorentz rho sigma beta t (v: float Vector) =
    let x = v[0]
    let y = v[1]
    let z = v[2]
    [ sigma * (y - x); x * (rho - z) - y; x * y - beta * z ] |> vector

let (rho, sigma, beta) = 28.0, 10.0, (8.0 / 3.0) // Model parameters

let f = lorentz rho sigma beta // Close over the model parameters
let y0 = vector [ -8.0; 8.0; 27.0 ] // Initial condition

let T = 50.0 // Time horizon
let N = 5000 // Number of samples
let dt = T / float N

// Integrate over the model to form the trajectory from the initial value.
let soln () =
    RungeKutta.FourthOrder(y0, dt, T, N, f) |> matrix

// Plotting code hidden for brevity.

(*** hide ***)
soln ()
|> trace "Butterfly attractor" line
|> GenericChart.ofTraceObject true
|> GenericChart.setLayout layout
|> GenericChart.toChartHTML
(*** include-it-raw ***)

(*** hide ***)
module Matrix =
    module Extensions =

        /// The following code extendes the MathNet Matrix type to
        /// allow logical row and column indexing.
        ///
        /// No work has been done to benchmark performance.
        type DenseMatrix with

            /// Support for logical indexing using ints
            member this.Item
                with set (logical: bool array array) value =

                    if logical.Length <> this.RowCount || logical[0].Length <> this.ColumnCount then
                        failwith "Logical Indexes must have same shape"

                    do
                        for (rowIndex, row) in logical |> Seq.indexed do
                            for (colIndex, elem) in row |> Seq.indexed do
                                if elem = true then
                                    this.Item(rowIndex, colIndex) <- value

            member this.Item
                with get ((logicalRows: bool array), (logicalCols: bool array)) =
                    if logicalRows.Length <> this.RowCount then
                        failwith "Error logical rows must have same count"

                    if logicalCols.Length <> this.ColumnCount then
                        failwith "Error logical columns must have same count"

                    let rowIndices = { 0 .. this.RowCount - 1 }
                    let colIndices = { 0 .. this.ColumnCount - 1 }

                    // These next operations may not be efficient
                    let slicedRows =
                        Seq.zip rowIndices logicalRows
                        |> Seq.filter (fun (_, selected) -> selected = true)
                        |> Seq.map (fun (index, _) -> this[index, *])
                        |> DenseMatrix.ofRowSeq

                    Seq.zip colIndices logicalCols
                    |> Seq.filter (fun (_, selected) -> selected = true)
                    |> Seq.map (fun (index, _) -> slicedRows[*, index])
                    |> DenseMatrix.ofColumnSeq

            member this.Item
                with set ((logicalRows: bool array), colIndex) values =

                    if logicalRows.Length <> this.RowCount then
                        failwith "Error logical rows must have same count"

                    if logicalRows.Length < (values |> Seq.length) then
                        failwith "Error row count must equal values count"

                    let rowIndices = { 0 .. this.RowCount - 1 }

                    let selectedRows =
                        Seq.zip rowIndices logicalRows
                        |> Seq.filter (fun (_, selected) -> selected = true)
                        |> Seq.map (fun (index, _) -> index)

                    Seq.zip selectedRows values
                    |> Seq.iter (fun (rowIndex, value) -> this[rowIndex, colIndex] <- value)
(** 

The Sequential Thresholded Least Squares implementation code below is taken from this authors repository at [Convexity](https://github.com/astrobytez/Convexity) 
*)

open Matrix.Extensions

module Internal =
    let smallIndicies (X: float Matrix) lambda =
        [| for row in X.EnumerateRows() -> [| for elem in row -> if abs (elem) < lambda then true else false |] |]

    let pointWiseNot elems =
        [| for x in elems -> [| for y in x -> if y = true then false else true |] |]

    let enumerateColumns (X: bool array array) =
        [| for ii in 0 .. X[0].Length - 1 -> [| for row in X -> row[ii] |] |]

/// The Sequential Threshold Least Squares algorithm.
///
/// numIters: Number of times to regress terms.
/// lambda: Sparsity promoting threshold used to truncate small terms.
/// A: System matrix
/// B: Regressor matrix
///
/// Algorithm adapted from:
/// https://arxiv.org/pdf/1509.03580.pdf.
let stlsq numIters lambda (A: DenseMatrix) (B: DenseMatrix) =
    let allRows = Array.replicate A.RowCount true
    let mutable (Xi: DenseMatrix) = DenseMatrix.OfMatrix(A.Solve(B))

    for _ in 1..numIters do

        // Truncate small values toward zero
        let mutable smallIndices = Internal.smallIndicies Xi lambda
        Xi[smallIndices] <- 0.0

        // Regress dynamics onto remaining terms to find sparse Xi
        let activeIndices = Internal.pointWiseNot smallIndices

        for (colIndex, bigInds) in activeIndices |> Internal.enumerateColumns |> Seq.indexed do
            let Sq = A[allRows, bigInds]
            let neSol = Sq.Solve(B[*, colIndex])

            Xi[bigInds, colIndex] <- neSol

    Xi

(*** hide ***)

/// Adapted from:
///
/// Data Driven Science & Engineering: Machine Learning, Dynamical Systems, and Control
/// by S. L. Brunton and J. N. Kutz
/// Cambridge Textbook, 2019
/// Copyright 2019, All Rights Reserved
/// http://databookuw.com/
let poolData polyOrder useSine (xData: float Matrix) =
    let n = xData.RowCount
    let m = xData.ColumnCount

    let ( .* ) (u: float Vector) v = u.PointwiseMultiply v

    let out0 = Seq.replicate n 1.0 |> vector |> Seq.singleton

    let out1 = xData.EnumerateColumns()

    let out2 =
        if polyOrder >= 2 then
            seq {
                for i in 0 .. m - 1 do
                    for j in i .. m - 1 -> xData[*, i] .* xData[*, j]
            }
        else
            Seq.empty

    let out3 =
        if polyOrder >= 3 then
            seq {
                for i in 0 .. m - 1 do
                    for j in i .. m - 1 do
                        for k in j .. m - 1 -> xData[*, i] .* xData[*, j] .* xData[*, k]
            }
        else
            Seq.empty

    let out4 =
        if polyOrder >= 4 then
            seq {
                for i in 0 .. m - 1 do
                    for j in i .. m - 1 do
                        for k in j .. m - 1 do
                            for l in k .. m - 1 -> xData[*, i] .* xData[*, j] .* xData[*, k] .* xData[*, l]
            }
        else
            Seq.empty

    let out5 =
        if polyOrder >= 5 then
            seq {
                for i in 0 .. m - 1 do
                    for j in i .. m - 1 do
                        for k in j .. m - 1 do
                            for l in k .. m - 1 do
                                for m in l .. m - 1 ->
                                    xData[*, i] .* xData[*, j] .* xData[*, k] .* xData[*, l] .* xData[*, m]
            }
        else
            Seq.empty

    let out6 =
        if useSine then
            Seq.empty // TODO handle trig functions
        else
            Seq.empty

    Seq.concat [ out0; out1; out2; out3; out4; out5; out6 ]
    |> DenseMatrix.ofColumnSeq

let line' =
    let tmp = Line()
    tmp?width <- 3.0
    tmp?color <- "#009999"
    tmp

(** 

The model library is constructed with the poolData function which is not shown here.
The code for this can be found in the original source of the article [here](https://github.com/astrobytez/blogs/blob/main/Machine%20Learning%20-%20Part%202.fsx)

*)

let polyOrder = 5 // Highest order of polynomials in the candidate library.

// Get the model trajectory data - the real system state measurements.
let traj = soln ()

// Compute the model library - construct our measurement data
// This along with the derivatives is what we are to reconstruct the model from.
let X = traj |> poolData polyOrder false |> DenseMatrix.OfMatrix

// Here we are computing the derivatives from the known state data
// using the real model. This gives more accurate derivatives which greatly improves the solution.
let X' =
    seq { for ii in 0 .. X.RowCount - 1 -> lorentz rho sigma beta 0.0 traj[ii, *] }
    |> matrix
    |> DenseMatrix.OfMatrix

// Call the solver and solve the coefficients for the model using sparse regression.
// Here we use 10 iterations and a threshold value of 0.1
let coeffs = stlsq 10 0.10 X X'

printfn $"{coeffs}"
(*** include-output ***)

(**
The coefficients shown are an exact representation of the original model in this case. Each column is the state variables x, y, and z respectively.
The rows represent the amount of each term which is active in the output of the system.

In this case we have constructed the terms to be the set of polynomials up to order 5.

The reconstructed model is given by:

$$
\dot X = \Theta (X) \varXi
$$

Refer to Data Driven Science & Engineering: Machine Learning, Dynamical Systems, and Control by S. L. Brunton and J. N. Kutz Chapter 7.3 for more information on the methods.

Next we plot the reconstructed model and show they are the same.
*)

// The reconstructed model function for a given state input vector, x.
let lorentz' t (x: float Vector) =
    (poolData polyOrder false (x.ToRowMatrix()) * coeffs)[0, *]

// Setup and extrapolate the new model.
let f' = lorentz'

let soln' () =
    RungeKutta.FourthOrder(y0, dt, T, N, f') |> matrix

// Plotting code hidden for brevity

(*** hide ***)
soln' ()
|> trace "Butterfly attractor reconstructed" line'
|> GenericChart.ofTraceObject true
|> GenericChart.setLayout layout
|> GenericChart.toChartHTML
(*** include-it-raw ***)

(**
This output shows the model is exactly the same as the original model used to construct the data.

## Wrapping up
The derivatives for a real systems data may not be readily available or the data may be noisy in both spacial and temporal coordinates.
All of these factors would result in a degradation of the predicted model performance, or the model may not converge on a good solution at all.

In all these cases the engineer has to be pragmatic and cross validate the models against available data to ensure the models are performing to an acceptable level.

Thanks for reading.
*)

