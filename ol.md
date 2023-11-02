
I think this is a job for cases from the amsmath package

enter image description here

\documentclass{article}
\usepackage{amsmath}

\begin{document}
\[
    f(x)= 
\begin{cases}
    \frac{x^2-x}{x},& \text{if } x\geq 1\\
    0,              & \text{otherwise}
\end{cases}
\]
\end{document}