data Expr = I Int
          | B Bool
          | Add Expr Expr
          | Eq  Expr Expr

eval :: Expr -> Maybe (Either Int Bool)
eval (I n)       = Just (Left n)
eval (B n)       = Just (Right n)
eval _ = undefined       -- Exercise
