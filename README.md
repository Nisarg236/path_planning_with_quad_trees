# path_planning_with_quad_trees
This repository converts 2D occupancy grid into a quad tree and implements planning algorithms for mobile robots using the generated quad tree. Planning works much faster on these because each square represents an area with free space and the neighbouring squares are already connected in a graph.

below image shows an occupancy grid converted to a quad tree:

![image](https://github.com/Nisarg236/path_planning_with_quad_trees/assets/71684502/f48ea915-8392-4e4b-b657-17d1a07065f2)

![image](https://github.com/Nisarg236/path_planning_with_quad_trees/assets/71684502/c930286f-6fc6-49e1-9e28-bf8cf236693a)

Currently only the graph generation part is made, I will build other functionalities to use the generated path with move_base and update here.

Fork, star, and contribute if you feel it might be useful! 🚀🌟🤖...
