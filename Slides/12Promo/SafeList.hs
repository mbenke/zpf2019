{-# LANGUAGE GADTs #-}
module SafeList where

data Empty
data NonEmpty

data SafeList a b where
  Nil :: SafeList a Empty
  Cons :: a -> SafeList a b -> SafeList a NonEmpty

safeHead :: SafeList a NonEmpty -> a
safeHead (Cons x _) = x
-- The pattern match is exhaustive!
