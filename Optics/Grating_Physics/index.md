---
title: Scalar theory of grating diffraction
description: An interactive introduction to scalar optics, diffraction and physical gratings.
authors:
  - name: Ashwin Hariharan
site:
  hide_outline: true
  hide_authors: true
---

:::{div}
:class: textbook-hero
**AN INTERACTIVE OPTICS TEXTBOOK**

From optical fields to diffraction orders: build physical intuition through mathematical explanations, numerical experiments and interactive Python.

**Ashwin Hariharan**

[Begin with 0.0: Foundations of scalar optics](0_0_Foundations_of_Scalar_Optics.ipynb)
:::

## About this book

Scalar theory of grating diffraction introduces the ideas needed to understand how optical fields interact with apertures and periodic structures. The introductory chapter connects scalar fields and the Helmholtz equation to boundary conditions, propagation and diffraction orders, then compares ideal masks with physical absorbing gratings.

Read the explanations and figures at your own pace, or activate the Python examples to explore how the fields change. Mathematical derivations and worked examples accompany the physical pictures.

[Meet the author and read about the approach in the Preface](preface.md). For navigation, exercises and live examples, see the [Reader's Guide](reader-guide.md).

## Chapter 0 · Foundations of scalar optics

The eight lessons below form the introductory sequence. Their original numbering is retained throughout the book.

::::{div}
:class: textbook-chapter-list
:::{div}
:class: textbook-chapter
### Fields and wave equations

Establish the language of complex fields and the solutions allowed by the wave equation.

- [0.0 · Foundations of scalar optics](0_0_Foundations_of_Scalar_Optics.ipynb)
- [0.1 · Scalar fields and Helmholtz](0_1_Scalar_Fields_and_Helmholtz.ipynb)
- [0.2 · Solutions of the Helmholtz equation](0_2_Solutions_of_the_Helmholtz_Equation.ipynb)
:::

:::{div}
:class: textbook-chapter
### Boundaries and propagation

Connect prescribed fields and outgoing conditions with scalar diffraction and propagation.

- [0.3 · Boundary conditions and diffraction](0_3_Boundary_Conditions_and_Diffraction.ipynb)
- [0.4 · Propagation of scalar optical fields](0_4_Propagation_of_Scalar_Optical_Fields.ipynb)
:::

:::{div}
:class: textbook-chapter
### Physical gratings and diffraction orders

Follow a solved grating field into its complex diffraction coefficients and compare physical and ideal models.

- [0.5 · Physical gratings and the Helmholtz boundary-value problem](0_5_From_Helmholtz_BVP_to_Diffracted_Field.ipynb)
- [0.6 · From complex fields to diffraction orders](0_6_From_Complex_Fields_to_Diffraction_Orders.ipynb)
- [0.7 · Ideal binary masks versus physical absorbing gratings](0_7_Ideal_Binary_Masks_vs_Physical_Gratings.ipynb)
:::
::::

## About the project

The lessons are authored as Jupyter notebooks and rendered with MyST and Jupyter Book. Python helpers support the numerical examples and interactive visualizations. The build workflow executes copies of the notebooks and checks numerical examples, rendered mathematics, downloadable bundles and public output, while preserving the authoring notebooks and their saved outputs.

[Browse the public book sources](https://github.com/ahariharan-hub/grating-physics-book), consult the [Reader's Guide](reader-guide.md), or get the [downloadable Python experiments and local setup instructions](Downloads.md).
