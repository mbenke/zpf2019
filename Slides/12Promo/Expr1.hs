data Expr = I Int
          | Add Expr Expr

eval :: Expr -> Int
eval (I n)       = n
eval (Add e1 e2) = eval e1 + eval e2

