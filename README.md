# FractalGravity

The fractals used in this study are referred to as Julia Sets. A Julia set is defined topologically as the perimeter separating the points $z \in \mathbb{C}$ that either approach a finite point $p$ (Basin of attraction, $\text{Bas}(p)$) or diverge to infinity (Basin of infinity, $\text{Bas}(\infty)$) under the iterates of the map $f_c$, where $f_c(z) = z^2 + c$.

\begin{align}
\text{Bas}(p) &: \{ z_{0}\in \mathbb{C}\mid  f^{n}(p) \rightarrow p \} \\
\text{Julia Set} &: J(f_{c})=\partial \{ z_{0}\in \mathbb{C}\mid  f^{n}(p) \rightarrow p \} \\
\text{Bas}(\infty) &: \{z_{0}\in \mathbb{C}\mid \lim_{n\rightarrow \infty}|f_{c}^{n}(z_{0})| = \infty \}
\end{align}

The map $f_c$ is iterated a finite number of times over a grid of points in the complex plane saved in an array. The iteration number at which a point either converges to a point $p$ or diverges to infinity, referred to as the "escape time", is recorded and its value is reassigned to that entry in the matrix. The array is decomposed into two separate arrays, $\text{Bas}(\infty)$ \& $\text{Bas}(p)$. The escape times are assigned a color based on their magnitude, using a different color scheme for the inner and outer basins. The color assigned to each point in the grid is treated as the magnitude of a potential field $U$ at that point, such that a particle within the vicinity of the fractal will experience a force $F = - \nabla U$. This force is non-conservative thus the behavior of a particle no longer obeys the conservation of energy principle nor Newtonian mechanics.
