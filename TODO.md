# TODOs

### Bug fixes
- [x] Change debiasing from `inplace` to `copy`
- [x] PCA on A + B only, then apply the projection matrix on the third.

### Different debiasing techniques
- [x] Linear projection
- [ ] OSCaR - Orthogonal Subspace Correction and Rectification

### UI changes
- [x] Dim other points on hover
- [x] Align axis ranges for PCA and Two-means
- [x] Remove PCA picture
- [x] Toggle mean-line and purple points
- [ ] Remove points without removing the visualization of that word
- [ ] 3D PCA, future stuff
- [ ] Make the mean direction always horizontal and mean 1 on right and mean 2 on left


### Animation Tasks
- [x] Animate collapse to two-means
- [ ] Animate PCA as rotation and scaling (but from what to what?)
- [ ] Mean point in debiasing (right)
- [ ] Animate between one debiasing method to another to show difference between methods

### Miscellaneous
- Assumption on gender of name
- Projection on normal to max margin plane
- PCA with maximal alignment
- Check Viz for debiasing for what types of techniques they've used  

----

- Google slide deck shared among everyone 
- Protected class/seed sets (bias direction) may have more than 2 inputs
- Subspace method selection on the seedset line
- Saved screenshot of Vivek's interface 
- Oscar takes two subspaces and orthogonalizes it, so seed-set pair 1 and seed-set pair 2.
- Classification - linear classifier, take normal to separating hyperplane as gender direction.
  
- Linear projection with 4 subspace methods:
    1. Two-means (two seed inputs)
    2. PCA (one input, csv)
    3. PCA-paired (hypen + csv)
    4. Classification (two seed inputs)
- Evaluation set

- Hard debiasing, again with 4 subspace methods
    - Evaluation set
    - Equalize set, similar to PCA-paired
    - Hidden large neutral set - just a label that it is there
    - [Optional] Generating set
    
- INLP
    1. Classification set, two boxes
    2. Evaluation set
    3. Slider for number of iterations, between 1-40, default is 35
    4. Show which step you are at in the SVG
    
- OSCaR
    1. Space 1 - same 4 subspace methods
    2. Space 2 - same 4 subspace methods
    3. Evaluation set
    
- QOL: Linked zoom and pan

- Take dot product with subspace direction, that will be x coordinate
- X - gender direction = X', and then PCA of X', which will have orthogonal components to gender direction
- ^Rather than above, project X to null space of gender direction and then take PCA 
- Color theme accessible to color-blind people, plus shape encoding, triangle or square
- Hover over point, click to retain highlighting across all views
- Classifier method - sample grid, check sign of classifier, and color accordingly
- WEAT score and residual bias measures

- Submit the slide deck
_______________________________________

- Lighter shade for menu
- Show equalize set, show gender-specific words do not move
- Classification - two point along the normal vector and then draw the vector in 2-d space
- x-axis alignment
- evaluation set dialog for OSCaR, 1st direction, 2nd direction.
- Show direction vectors for two dirs in OSCaR
- Top two - align x-axis, show vectors
- preloaded examples

--------------------------------------
- [x] Linear projection - use same fitted PCA for base and debiased 
- [x] Add step to show subspace method computations
- [ ] Rotation for bias direction alignment instead of flipping
- [x] Visual indication for camera movement vs actual projections
- [x] gendered words instead of names, use names in the critiques of debiasing
- [x] Bias direction projection
- [x] Checkbox for toggle buttons
- [x] Do not remove origin bias labels on toggle data labels
- [ ] WEAT score in initial and debiased embedding
- [x] Not show all equalize set
- [x] Bias1 and Bias 2 -> ~concept 1 and concept 2~; subspace 1 and subspace 2 
- [x] ability to name the subspaces
- [x] Update v to v' in Oscar step
- [x] Disable other subspace methods
- [x] INLP - always reorient to classifier at each step, should be two steps - reorient to classifier direciton and project away from it
- [x] Scale of x and y-axis be same, balance +-x and +-y, animate axes
- [x] Slow animation
- [ ] Rotation using dynamic projections

-----------------------------------

- [x] Toggle buttons
- [x] Camera movement indicator
- [x] Custom names for subspaces
- [x] INLP break down classifier finding and rotation step

----------------------------------

- [x] INLP, 4 steps
- [x] Fix Oscar
- [x] Linear projection, Hard debiasing: concept vector shrink to origin
- [x] INLP, stop when score < 0.5 + epsilon, atleast two rounds
- [x] Fix flicker-class camera
- [x] move to preloaded examples
- [x] For axes - same-scale, symmetric, square around origin
- [x] Zoom for initial and debiased embedding
- [x] Fixed set of algo+subspace method
- [x] Combinations: Linear + Hard + Oscar - Two Means + PCA; INLP - classification
- [x] On error, dialog box
- [ ] Support ~empty~ evalset, single word evaluation
- [x] Linear projection - Linear projection
- [x] Correct remove points
- [x] Dotted line for classifier
- [x] lose focus in text field during overlay

--------------------------------

- [x] Label placement to avoid overlapping labels 
- [x] Bug fixes involving zooming during animation and resetting the initial embedding: including disabling the zoom when the points are moving in the Embedding View panel.
- [x] Export as CSV
- [ ] Add 'Save Embeddings' and 'Import Embeddings' buttons on the Control Panel
- [ ] Show WEAT scores before and after debiasing for a fixed set of words
- [ ] When 'data labels' are turned off, allow the evaluation set labels to be turned back on. More general functionality: allow a user-clicked subset of points with the labels turned on during the animation.
- [x] Show k=10 nearest neighbor as a list from the high-dimensional embedding.

- Oscar extract n-2 space and stack with 2-d for neighbor and csv queries
- VERB /newline Visualization...
- Save as SVG js library
- Oscar two labels, PCA 50-d view first in oscar
- PCA-paired, hard-debiasing hyphenated list
