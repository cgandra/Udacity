# Project: Implement Route Planner

## Correctness

| Success Criteria                                      | Specifications  |
| ----------------------------------------------------- | --------------- |
| Does the submission pass all tests?                   |  Running test.py shows "all tests pass". |
| Does student implement A* search methods?             | The student implements all required methods.| 
| Is the heuristic used in A* search admissible?        | The heuristic function used to estimate the distance between two intersections is guaranteed to return a distance which is less than or equal to the true path length between the intersections.| 
| Did Student answer all of the questions correctly?    | Student answered all question correctly.| 


## Choice and Usage of Data Structures

| Success Criteria                                      | Specifications  |
| ----------------------------------------------------- | --------------- |
| Does code avoid slow lookups through correct choice of data structures? | Code avoids obvious inappropriate use of lists and takes advantage of the performance improvement afforded by sets / dictionaries where appropriate. For example, a data structure like the "open_set" on which membership checks are frequently performed (e.g. if node in open_set) should not be a list. |
| Does code avoid flagrant unnecessary performance problems? | This item is a judgement call. Student code doesn't need to be perfect but it should avoid big performance degrading issues like......unnecessary duplication of lists ...looping through a large set or dictionary when a single constant-time lookup is possible|