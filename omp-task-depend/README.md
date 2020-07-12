Attempts to use task dependencies to interleave all three stages of tree traversals, upward, dual tree traversal, downward.

However, the OpenMP spec limits the implementation:

- Can't declare task dependencies on members of structs
- Can't declare task dependencies on 2D ranges (e.g. for all points in all children nodes of the current node)
- No commutative task dependency (yet)
