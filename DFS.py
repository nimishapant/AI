# Name: Nimisha Pant

goal_state = [1,2,3,4,5,6,7,8,0]

moves = {
    0: [1,3], 1: [0,2,4], 2: [1,5],
    3: [0,4,6], 4: [1,3,5,7], 5: [2,4,8],
    6: [3,7], 7: [4,6,8], 8: [5,7]
}

def dfs(start):
    stack = [(start, [])]
    visited = set()

    while stack:
        state, path = stack.pop()

        if tuple(state) in visited:
            continue

        visited.add(tuple(state))

        if state == goal_state:
            return path + [state]

        zero = state.index(0)

        for move in moves[zero]:
            new_state = state.copy()
            new_state[zero], new_state[move] = new_state[move], new_state[zero]
            stack.append((new_state, path + [state]))

    return None


start_state = [1,2,3,4,0,5,6,7,8]

result = dfs(start_state)

if result:
    for step in result:
        print(step)
else:
    print("No solution found")