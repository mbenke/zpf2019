{-# LANGUAGE 
    LambdaCase, GADTs, TypeOperators, TypeFamilies, DataKinds #-}
-- https://gist.github.com/AndrasKovacs/9229103

data Type = TInt | TBool | Type :=> Type

-- Needs GHC >= 7.8
type family Interp (t :: Type) where
    Interp TInt      = Int
    Interp TBool     = Bool 
    Interp (a :=> b) = Interp a -> Interp b

data Var (cxt :: [Type]) (t :: Type) where
    ZVar :: Var (t ': ts) t 
    SVar :: Var ts t -> Var (t2 ': ts) t

data Expr (cxt :: [Type]) (t :: Type) where
    Val :: Int -> Expr cxt TInt
    Var :: Var cxt t -> Expr cxt t 
    Lam :: Expr (a ': cxt) t -> Expr cxt (a :=> t)
    Op  :: (Interp a -> Interp b -> Interp c) -> Expr cxt a -> Expr cxt b -> Expr cxt c
    App :: Expr cxt (a :=> b) -> Expr cxt a -> Expr cxt b
    If  :: Expr cxt TBool -> Expr cxt a -> Expr cxt a -> Expr cxt a

data Env (cxt :: [Type]) where
    ZEnv :: Env '[]
    SEnv :: Interp t -> Env ts -> Env (t ': ts)

lookupVar :: Var cxt t -> Env cxt -> Interp t 
lookupVar ZVar     (SEnv x xs) = x
lookupVar (SVar i) (SEnv _ xs) = lookupVar i xs
-- lookupVar _        _           = undefined

eval :: Env cxt -> Expr cxt t -> Interp t
eval env = \case
    Val i    -> i
    Op f a b -> f (eval env a) (eval env b)
    Var i    -> lookupVar i env
    Lam f    -> \x -> eval (SEnv x env) f
    App f a  -> eval env f $ eval env a
    If p a b -> if eval env p then eval env a else eval env b

eval' :: Expr '[] t -> Interp t
eval' = eval ZEnv

vs (Var n) = Var (SVar n)
v0 = Var ZVar 
v1 = vs v0
v2 = vs v1

type Cxt1 = TInt : '[]

ibody :: Expr Cxt1 TInt
ibody = v0

i = Lam ibody
i42 = App i (Val 42)
