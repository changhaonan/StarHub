# StarHub

A unified library shared by all Star-related projects.

## Plan

- [x] Add measurement module.

### Oct.24th

- [x] Move torch module, optical-flow.

### Oct.26th

- [x] Processor structure and etc.
    - [x] Adapt NodeGraph to the system.
    - [x] Move Renderer
    - [x] Finish the Measurement pipeline

### Oct.27th

- [ ] Optimization structure.
    - [ ] Move Geometry Initialization structure.
    - [ ] Add Visualization structure.

### Oct.29th

- [ ] Joint testing.

## Problem

Can not be statically determinted. What is the problem??

## Philosophy

I want to seperate the star lib and star app. The star app should have shareable structure. For example, I can have different apps but share the same lib.