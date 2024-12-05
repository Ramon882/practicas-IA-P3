class Node:
    # Estructura básica de un nodo de árbol binario
    def __init__(self, key):
        self.left = None
        self.right = None
        self.val = key

def insert(root, key):
    # Insertar un nuevo nodo en el árbol
    if root is None:
        return Node(key)
    if key < root.val:
        root.left = insert(root.left, key)
    else:
        root.right = insert(root.right, key)
    return root

def inorder_traversal(root, res):
    # Realizamos un recorrido en orden para obtener los valores ordenados
    if root:
        inorder_traversal(root.left, res)
        res.append(root.val)
        inorder_traversal(root.right, res)

def tree_sort(arr):
    # Inicializamos el árbol binario
    if not arr:
        return []
    root = Node(arr[0])
    for i in range(1, len(arr)):
        insert(root, arr[i])
    result = []
    inorder_traversal(root, result)
    return result

# Ejemplo de uso
arr = [64, 34, 25, 12, 22]
sorted_arr = tree_sort(arr)
print("Ordenado:", sorted_arr)
