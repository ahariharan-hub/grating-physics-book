---
title: Preface
description: The motivation, author, and educational philosophy behind the book.
authors:
  - name: Ashwin Hariharan
---

## Welcome

*Scalar theory of grating diffraction* explores how optical fields interact with apertures and periodic structures. A grating may look simple, but connecting its structure to the field it produces brings together wave propagation, boundary conditions and interference.

This book approaches those ideas through mathematics and computation. An equation can tell us what a model predicts; a calculation and a figure help us ask what that prediction means. The motivation is to move between those views until the physics becomes easier to reason about.

## A little about me

I'm an engineer working at **ASML**, with a curiosity for physics and a particular fascination with optics.

I enjoy **learning, teaching, and coding**—preferably all three at the same time.

My favorite way to understand something is to turn the mathematics into code, make a plot, move a slider, and see what happens. Sometimes the physics behaves exactly as expected. Sometimes it has other plans.

I believe that physics becomes much more intuitive when we can **see it in action**. That's the philosophy behind these notebooks: exploring concepts through mathematical reasoning, numerical experiments, and interactive visualizations.

I'm learning along the way, and this book is my way of sharing that journey.

After all, what's the fun in understanding something if you can't share it?

## The approach

Three complementary perspectives run through the lessons:

- **Physical intuition:** What is the field doing, and what behaviour should we expect before calculating anything?
- **Mathematical description:** Which equations, boundary conditions and assumptions describe the problem, and how do their solutions connect to the physical picture?
- **Numerical exploration:** What happens when we calculate the field, vary a parameter or compare two models?

Each perspective helps question the others. A plot can reveal behaviour that is hard to see in an expression; the mathematics can explain it; physical reasoning can suggest whether either result makes sense. The interactive examples are invitations to investigate those connections, rather than pictures to accept at face value.

## Who is this book for?

This book is for students, engineers and curious readers interested in wave optics and grating diffraction. It may serve as a companion to formal study or as a place to revisit familiar ideas through computation.

Familiarity with complex numbers, differentiation and integration is helpful. Differential equations and Fourier ideas become useful as the discussion develops. You do not need to write Python to read the explanations and figures; some programming experience helps when you want to adapt an experiment.

## A few words before you begin

Be curious, make predictions and try things. Change a parameter and ask why the result moved—or why it did not. When an answer is available, give yourself a chance to reason through the question before opening it.

Keep a critical eye on the models as well as the plots. Scalar theory makes assumptions about the field, and numerical calculations introduce choices about sampling, resolution and convergence. A smooth figure is not, by itself, evidence that a calculation is accurate or that a model applies to a particular physical grating.

An unexpected result is a useful starting point: check the units, revisit the assumptions and see whether the behaviour survives a more careful calculation. Understanding often begins with a result that refuses to cooperate.

For navigation, exercises and interactive sessions, continue to the [Reader's Guide](reader-guide.md). Or start with [0.0: Foundations of scalar optics](0_0_Foundations_of_Scalar_Optics.ipynb).
