This code traces the vasculature of a frog tongue as sketched in Conheim's seminal experiments on the blood vessels of a frog tongue:

Input:
<p align="center">
  <img src="https://github.com/IngeborgGjerde/frog-tongue-graph/blob/main/Reproduction-of-Cohnheims-seminal-experimental-work-on-the-embolic-process-in-the-frog.png">
</p>

The code traces the arteries (red) and veins (blue). A skeleton is formed for each, and floodfill is used to mark the different branches. 

Output:
<p align="center">
  <img src="https://github.com/IngeborgGjerde/frog-tongue-graph/blob/main/vasculature.jpg">
</p>

## Dependencies
Skimage and opencv

## Todo
- Go from skeleton to graph.
