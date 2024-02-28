(*** hide ***)

(*** condition: prepare ***)
#r "nuget: Plotly.NET, 4.2.0"
#r "nuget: Plotly.NET.Interactive, 4.2.0"

open System
open Plotly.NET

Plotly.NET.Defaults.DefaultDisplayOptions <-
    Plotly.NET.DisplayOptions.init (PlotlyJSReference = Plotly.NET.PlotlyJSReference.NoReference)

(**
# Machine Learning - part 1

This is part 1 of a series looking in the fundamentals of machine learning.

The series aims to be familiar to readers who have completed a first year university math course.

### Introduction - Optimisation is a cornerstone of Machine Learning

But what is optimisation? Optimisation is the process of trying to find the best of something. However in order to find the "best" of something we have to have a way to compare
said things. In mathematical terms the way in which this is done is by forming a "Cost Function". The cost function is a function which
assigns a value to the given inputs for the objective.

For example a general form cost function could be:

$$
J = f(u, p, t)
$$

This equation describes a function of inputs $u$, parameters $p$ and time $t$.

We can then perform a variety of searching methods to look at the inputs (search space) and see which give the lowest cost(s). These become the "best" values which we are looking for.
This process of optimisation is the cornerstone of most machine learning algorithms. Somewhere inside most machine learning algorithms you will find an optimisation problem being solved in this way.

In fact most algorithms will have a cost function which looks to minimise errors between given observations and predictions. Later in this series we will look at this in detail.

### Minimisation and Root Finding

In some fictional scenario we can create a cost function for something we wish to optimise, for example a simple quadratic function $f(x) = x^2 - 3$.
How would you find where the function crosses the x-axis? How can we find the global minimum?

First lets plot the functions and inspect visually.


*)

#r "nuget: Plotly.NET, 4.2.0"

open Plotly.NET
open Plotly.NET.LayoutObjects
open StyleParam

let x1Data = [ -5.0..0.05..5.0 ]
let y1Data = x1Data |> List.map (fun x -> x ** 2 - 3.0)
let y1DotData = x1Data |> List.map (fun x -> 2.0 * x)

let config = Config.init (Responsive = true, TypesetMath = true)

let layout = Layout.init (ShowLegend = false)

let chart1 =
    Chart.Line(x1Data, y1Data, Name = "Quadratic")
    |> Chart.withXAxisStyle ("xAxis")
    |> Chart.withYAxisStyle ("yAxis")

let chart2 =
    Chart.Line(x1Data, y1DotData, Name = "Linear - derivative")
    |> Chart.withXAxisStyle ("xAxis")
    |> Chart.withYAxisStyle ("yAxis")

let chart3 =
    Chart.Point([ - sqrt(3.0); 0.0; + sqrt(3.0) ], [ 0.0; 0.0; 0.0 ], Name = "Roots")
    |> Chart.withXAxisStyle ("xAxis")
    |> Chart.withYAxisStyle ("yAxis")

Chart.combine [ chart1; chart2; chart3 ]
|> Chart.withConfig config
|> Chart.withLayout layout
|> GenericChart.toChartHTML
(*** include-it-raw ***)

(**

We can solve for the roots manually (crosses x-axis) by using algebra:
$$
\displaylines{
\text{let } f(x) = 0 \\
0 = x^2 - 3 \\
x^2 = 3 \\
x = \sqrt{3} \\
\therefore x = \pm 1.73205
}
$$

We can solve the minimum (optimal x gives lowest cost) by taking the derivative and setting it to zero:
$$
\displaylines{
\text{let } J = f(x) = x^2 - 3 \\
f(x) = x^2 - 3 \\
f'(x) = 2x \\
\text{Set f'(x) to zero and solve} \\
2x = 0 \\
\therefore x = 0
}
$$

Here we see the minimum for this particular function occurs when $x$ is zero.

### Numerical Techniques

Some elementary functions including cubics like the problem we explore next can be solved analytically. However, this does not hold in general more complex functions require numerical 
techniques to find the solutions.

For a simple yet motivating example we will solve the following cubic polynomial equation:

$$
f(x) = -0.3x^3 + 2.x^2 + 0.5x - 4.0
$$

> Note there is more than one root, this is important in numerical approaches since the initial guess will affect the root which is found if any.

As can be seen in the following plot of the equation, there are three *"roots"* or *"solutions"*.
There is also a local minima and maxima.

---

*)

let objective x =
    -0.3 * x ** 3.0 + 2.0 * x ** 2.0 + 0.5 * x - 4.0

let x2Data = [ -3.0..0.05..7.0 ]
let y2Data = x2Data |> List.map objective

let chart4 =
    Chart.Line(x2Data, y2Data, Name = "Cubic")
    |> Chart.withXAxisStyle ("xAxis")
    |> Chart.withYAxisStyle ("yAxis")

let chart5 =
    Chart.Point([ -1.394; +1.447; +6.614 ], [ 0.0; 0.0; 0.0 ], Name = "Roots")
    |> Chart.withXAxisStyle ("xAxis")
    |> Chart.withYAxisStyle ("yAxis")

Chart.combine [ chart4; chart5 ]
|> Chart.withConfig config
|> Chart.withLayout layout
|> GenericChart.toChartHTML
(*** include-it-raw ***)

(**

### Fundamental limit theorem

We can use concepts of calculus to find the roots and we will use numerical methods
to find the roots programatically.

The [fundamental limit theorem](https://en.wikipedia.org/wiki/Fundamental_theorem_of_calculus) provides the basis for the idea of finding roots.

The main outcome of the theorem is the following equation:

$$
f'(x) = \lim_{h\to 0} \frac{f(x + h) - f(x)}{h}
$$

This is also called the forward difference derivative.

In practice it can be shown the computation using the fundamental limit theorem suffers from floating point cancellation.

To make things more interesting, albeit unnecessary, we will show an alternative method to correct for the errors in forward difference called complex step.

---

### Complex Step

Where the function $f$ supports the domain of Complex numbers, an innovative solution to the numerical 
issues of the forward difference method is called
[Complex Step](https://mdolab.engin.umich.edu/wiki/guide-complex-step-derivative-approximation).

This method can be used to improve the numerical accuracy and stability of the derivative. 
*)

open System
open System.Numerics

/// Complex step method to compute derivatives.
///
/// Here we use the step size h of 1e-10.
let complexStep f x =
    let h = 1e-10
    let num: Complex = f (x + Complex(0.0, h))
    Complex(num.Imaginary / h, 0.0)

(**
In order to find the roots of the system we can use calculus to solve the following problem: $f(x) = 0$, that is;

$$
f(x) = -0.3x^3 + 2.x^2 + 0.5x - 4.0 = 0
$$

The [newton-raphson](https://en.wikipedia.org/wiki/Newton%27s_method) is a method for recursively improving the solution to the root of the equation.
It works by using the tangents of the function to move closer to the root.

*)

/// Newton Raphson method to compute the roots of a function
///
/// calcDeriv is the function to compute the first-order numerical
/// derivative of the function applied to the point x0.
///
/// When the maximum iterations have elapsed or the solution is within tolerance
/// the solution is done and the function will return the result.
let newtonRaphson calcDeriv maxIter tolerance (f: Complex -> Complex) x0 =
    let rec loop xn iter =
        if iter >= maxIter || Complex.Abs(f xn) < tolerance then
            xn
        else
            // Newton update step
            let xn1 = xn - f xn / calcDeriv f xn
            loop xn1 (iter + 1)

    loop x0 0

(**
---

### Building a Solver

In order to solve for the roots we need to setup the input problem and
create a solver.


*)

/// Provides a Solver Interface for finding roots of functions.
module Solver =

    ///
    /// Private helper to set the root and the method.
    let private solver = newtonRaphson complexStep 10 1e-6

    /// The public solver function
    ///
    /// Takes the function we want to solve,
    /// and an initial guess.
    /// The given function 'f' must support the interface: Complex -> Complex
    let findRoots (f: Complex -> Complex) x0 =
        let result = solver f (Complex(x0, 0.0))
        result.Real

(**
---

### Solution

Now we have the solver we can compute the solution for the problem.

*)
// Here we re-define the objective with complex
// types to be compatible with the complex solver.

let f (x: Complex) =
    -0.3 * x ** 3.0 + 2.0 * x ** 2.0 + 0.5 * x - 4.0

printfn $"The 1st root is %+.3f{Solver.findRoots f +1.0}"
printfn $"The 2nd root is %+.3f{Solver.findRoots f -1.0}"
printfn $"The 3rd root is %+.3f{Solver.findRoots f +5.0}"
(*** include-output ***)

(**
Note how the different initial conditions result in different outcomes. The algorithm will converge toward a solution 'downhill' from the initial condition.

*)

// We will also inspect the plot of the derivative

let x3Data = x2Data

let y3Data =
    x3Data
    |> List.map (fun x -> complexStep f (Complex(x, 0.0)))
    |> List.map (fun x -> x.Real)

let chart6 =
    Chart.Line(x3Data, y3Data, Name = "Derivative")
    |> Chart.withXAxisStyle ("xAxis")
    |> Chart.withYAxisStyle ("yAxis")

Chart.combine [ chart4; chart6 ]
|> Chart.withConfig config
|> Chart.withLayout layout
|> GenericChart.toChartHTML
(*** include-it-raw ***)

(**

Next we will create a test using Math.NET Numerics to validate these solutions.

---

### Testing

To test the results are correct we can use Math.NET Numerics to provide a reference to test against.
Math.Net provides methods for solving roots in the general case, however also provides a method for solving Cubic Polynomials, which is
our function in this case. Therefore we will select the [Cubic](https://numerics.mathdotnet.com/api/MathNet.Numerics.RootFinding/Cubic.htm) method for the test. First we must normalise the coefficients in the form:

$$
x^3 + a_2 x^2 + a_1 x + a_0 = 0
$$

*)

#r "nuget: MathNet.Numerics, 5.0.0"
open MathNet.Numerics.RootFinding

// A helper to convert list of 3 elems to a tuple
let tuple3 (coll: 'a list) = coll[0], coll[1], coll[2]

// Convert the coefficients to normal form, reverse them and pick the first three
// to be consistent with the API for the Math.NET Numerics Cubic function.
let coeffs = [ -0.3; 2.0; 0.5; -4.0 ]

let normal =
    coeffs
    |> List.map (fun x -> x / (Seq.head coeffs))
    |> List.rev
    |> List.take 3
    |> tuple3

let (root3, root2, root1) = Cubic.RealRoots(normal).ToTuple()

printfn $"The 1st root is %+.3f{root1}"
printfn $"The 2nd root is %+.3f{root2}"
printfn $"The 3rd root is %+.3f{root3}"
(*** include-output ***)

(**

These results agree with what we have from the newton method solution. We can also validate the results by inspection of the plots above.

---

### Conclusion

We have looked at optimisation being a fundamental part of the machine learning process. We looked at how to find roots of a smooth differentiable function using newtons method.

Thanks for reading!
*)
