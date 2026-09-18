import std/[algorithm, math, sequtils, strutils]

type
  Point* = object
    coords*: seq[float]
  
  KDNode*[T] = ref object
    point*: Point
    data*: T
    left*: KDNode[T]
    right*: KDNode[T]
    axis*: int
  
  KDTree*[T] = object
    root*: KDNode[T]
    dimensions*: int

# Point constructors and utilities
proc newPoint*(coords: varargs[float]): Point =
  Point(coords: @coords)

proc newPoint*(coords: seq[float]): Point =
  Point(coords: coords)

proc `[]`*(p: Point, i: int): float =
  p.coords[i]

proc `[]=`*(p: var Point, i: int, val: float) =
  p.coords[i] = val

proc len*(p: Point): int =
  p.coords.len

proc `$`*(p: Point): string =
  "(" & p.coords.mapIt($it).join(", ") & ")"

proc distanceSquared*(a, b: Point): float =
  ## Calculate squared Euclidean distance between two points
  result = 0.0
  for i in 0 ..< a.coords.len:
    let diff = a.coords[i] - b.coords[i]
    result += diff * diff

proc distance*(a, b: Point): float =
  ## Calculate Euclidean distance between two points
  sqrt(distanceSquared(a, b))

# KD-Tree operations
proc newNode[T](point: Point, data: T, axis: int): KDNode[T] =
  KDNode[T](
    point: point,
    data: data,
    left: nil,
    right: nil,
    axis: axis
  )

proc buildKDTree*[T](points: seq[(Point, T)], depth: int = 0): KDNode[T] =
  ## Build a balanced kd-tree from a sequence of points
  if points.len == 0:
    return nil
  
  let k = points[0][0].len  # number of dimensions
  let axis = depth mod k
  
  # Sort points by the current axis
  var sortedPoints = points
  sortedPoints.sort(proc(a, b: (Point, T)): int =
    cmp(a[0][axis], b[0][axis])
  )
  
  let median = sortedPoints.len div 2
  let (medianPoint, medianData) = sortedPoints[median]
  
  result = newNode(medianPoint, medianData, axis)
  result.left = buildKDTree(sortedPoints[0 ..< median], depth + 1)
  result.right = buildKDTree(sortedPoints[median + 1 .. ^1], depth + 1)

proc newKDTree*[T](points: seq[(Point, T)]): KDTree[T] =
  ## Create a new kd-tree from points with associated data
  if points.len == 0:
    return KDTree[T](root: nil, dimensions: 0)
  
  let dimensions = points[0][0].len
  KDTree[T](
    root: buildKDTree(points),
    dimensions: dimensions
  )

# Nearest neighbor search
proc nearestNeighborRec[T](
  node: KDNode[T],
  target: Point,
  best: var tuple[point: Point, data: T, dist: float]
): void =
  if node == nil:
    return
  
  # Check current node
  let dist = distanceSquared(node.point, target)
  if dist < best.dist:
    best = (node.point, node.data, dist)
  
  let axis = node.axis
  let diff = target[axis] - node.point[axis]
  
  # Choose which side to search first
  let (first, second) = if diff < 0:
    (node.left, node.right)
  else:
    (node.right, node.left)
  
  # Search the closer side first
  nearestNeighborRec(first, target, best)
  
  # Check if we need to search the other side
  if diff * diff < best.dist:
    nearestNeighborRec(second, target, best)

proc nearestNeighbor*[T](tree: KDTree[T], target: Point): tuple[point: Point, data: T, distance: float] =
  ## Find the nearest neighbor to the target point
  if tree.root == nil:
    raise newException(ValueError, "Cannot search empty tree")
  
  var best = (point: tree.root.point, data: tree.root.data, dist: Inf)
  nearestNeighborRec(tree.root, target, best)
  
  result = (point: best.point, data: best.data, distance: sqrt(best.dist))

# K-nearest neighbors search
proc kNearestNeighborsRec[T](
  node: KDNode[T],
  target: Point,
  k: int,
  heap: var seq[tuple[point: Point, data: T, dist: float]]
): void =
  if node == nil:
    return
  
  let dist = distanceSquared(node.point, target)
  
  if heap.len < k:
    heap.add((node.point, node.data, dist))
    heap.sort(proc(a, b: tuple[point: Point, data: T, dist: float]): int =
      cmp(b.dist, a.dist))
  elif dist < heap[^1].dist:
    heap[^1] = (node.point, node.data, dist)
    heap.sort(proc(a, b: tuple[point: Point, data: T, dist: float]): int =
      cmp(b.dist, a.dist))
  
  let axis = node.axis
  let diff = target[axis] - node.point[axis]
  
  let (first, second) = if diff < 0:
    (node.left, node.right)
  else:
    (node.right, node.left)
  
  kNearestNeighborsRec(first, target, k, heap)
  
  if heap.len < k or diff * diff < heap[^1].dist:
    kNearestNeighborsRec(second, target, k, heap)

proc kNearestNeighbors*[T](tree: KDTree[T], target: Point, k: int): seq[tuple[point: Point, data: T, distance: float]] =
  ## Find the k nearest neighbors to the target point
  if tree.root == nil:
    raise newException(ValueError, "Cannot search empty tree")
  
  if k <= 0:
    return @[]
  
  var heap: seq[tuple[point: Point, data: T, dist: float]] = @[]
  kNearestNeighborsRec(tree.root, target, k, heap)
  
  result = heap.mapIt((point: it.point, data: it.data, distance: sqrt(it.dist)))
  result.reverse()

# Range search
proc rangeSearchRec[T](
  node: KDNode[T],
  center: Point,
  radius: float,
  radiusSq: float,
  results: var seq[tuple[point: Point, data: T, distance: float]]
): void =
  if node == nil:
    return
  
  let dist = distanceSquared(node.point, center)
  if dist <= radiusSq:
    results.add((node.point, node.data, sqrt(dist)))
  
  let axis = node.axis
  let diff = center[axis] - node.point[axis]
  
  # Search both sides if needed
  if diff - radius <= 0:
    rangeSearchRec(node.left, center, radius, radiusSq, results)
  if diff + radius >= 0:
    rangeSearchRec(node.right, center, radius, radiusSq, results)

proc rangeSearch*[T](tree: KDTree[T], center: Point, radius: float): seq[tuple[point: Point, data: T, distance: float]] =
  ## Find all points within a given radius of the center point
  if tree.root == nil:
    return @[]
  
  let radiusSq = radius * radius
  rangeSearchRec(tree.root, center, radius, radiusSq, result)

# Example usage and tests
when isMainModule:
  import random
  
  # Create sample data
  randomize()
  var points: seq[(Point, string)] = @[]
  
  # Generate random 2D points
  for i in 0 ..< 20:
    let x = rand(100.0)
    let y = rand(100.0)
    points.add((newPoint(x, y), "point_" & $i))
  
  # Build the kd-tree
  let tree = newKDTree(points)
  
  echo "KD-Tree built with ", points.len, " points"
  echo "Dimensions: ", tree.dimensions
  echo()
  
  # Test nearest neighbor
  let target = newPoint(50.0, 50.0)
  let nearest = tree.nearestNeighbor(target)
  echo "Nearest neighbor to ", target, ":"
  echo "  Point: ", nearest.point
  echo "  Data: ", nearest.data
  echo "  Distance: ", nearest.distance
  echo()
  
  # Test k-nearest neighbors
  let k = 3
  let kNearest = tree.kNearestNeighbors(target, k)
  echo k, " nearest neighbors to ", target, ":"
  for i, neighbor in kNearest:
    echo "  ", i+1, ". ", neighbor.point, " (", neighbor.data, ") - distance: ", neighbor.distance
  echo()
  
  # Test range search
  let radius = 20.0
  let inRange = tree.rangeSearch(target, radius)
  echo "Points within radius ", radius, " of ", target, ":"
  for point in inRange:
    echo "  ", point.point, " (", point.data, ") - distance: ", point.distance