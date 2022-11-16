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

### Oct.29th

- [x] Initialization.
    - [x] Move Geometry Initialization structure.
    - [x] Add Visualization structure.

### Nov.15th

- [x] Make opticalflow working.

### Nov. 16th ~ 17th

- [ ] Make the optimization working.
- [ ] Unify SolverMaps & ObservationMaps into SurfelMapTex
 
### Nov. 18th

- [ ] Combined optimization.

## Problem

Can not be statically determinted. What is the problem??

## Philosophy

I want to seperate the star lib and star app. The star app should have shareable structure. For example, I can have different apps but share the same lib.