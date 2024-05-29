# path_planning_with_quad_trees
This repository converts 2D occupancy grid into a quad tree and implements planning algorithms for mobile robots using quad tree. Planning works much faster on these because each square represents an area with free space and the neighbouring squares are already connected in a graph.

below image shows an occupancy grid converted to a quad tree:
![image](https://github.com/Nisarg236/path_planning_with_quad_trees/assets/71684502/52a81d1b-fc87-499e-8341-27946844bf47)

Currently only the graph generation part is made, will build other functionalities to use the generated with move_base and update here.

Fork, star, and contribute if you feel it might be useful! 🚀🌟🤖...
