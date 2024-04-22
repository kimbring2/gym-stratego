letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
numbers = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']


pool_positions = ['h5', 'g5', 'd5', 'c5', 'h6', 'g6', 'd6', 'c6']

def create_stratego_labels():
    labels_array = []
    for l1 in range(10):
        for n1 in range(10):
            '''
            # For normal unit movement
            [(l1 + a, n1) for a in [-1, 1]]
            [(l1, n1 + b) for b in [-1, 1]]

            # For scount unit movement
            [(t, n1) for t in range(10)]
            [(l1, t) for t in range(10)]
            '''
            destinations = [(l1 + a, n1) for a in [-1, 1]] + [(l1, n1 + b) for b in [-1, 1]] + \
                           [(t, n1) for t in range(10)] + [(l1, t) for t in range(10)]
            
            for (l2, n2) in destinations:
                if (l1, n1) != (l2, n2) and l2 in range(10) and n2 in range(10):
                    #print("letters[l1] + numbers[n1]: ", letters[l1] + numbers[n1])
                    ego = letters[l1] + numbers[n1]
                    destination = letters[l2] + numbers[n2]
                    if ego in pool_positions or destination in pool_positions:
                        #print("ego: ", ego)
                        #print("destination: ", destination)
                        #print("")
                        continue

                    move = ego + destination
                    labels_array.append(move)

    return labels_array