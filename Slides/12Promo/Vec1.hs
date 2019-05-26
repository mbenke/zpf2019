{-# LANGUAGE GADTs #-}
{-# LANGUAGE DataKinds, KindSignatures #-}

module Vec1 where

data Nat :: * where
  Z :: Nat
  S :: Nat -> Nat

-- This defines
-- Type Nat
-- Value constructors: Z, S

-- Promotion (lifting) to type level yields
-- kind Nat
-- type constructors: 'Z :: Nat; 'S :: Nat -> Nat
-- 's can be omitted in most cases, but...

-- data P          -- 1
-- data Prom = P   -- 2
-- type T = P      -- 1 or promoted 2?
-- quote disambiguates:
-- type T1 = P     -- 1
-- type T2 = 'P    -- promoted 2

data Vec :: Nat -> * -> * where
  Vnil :: Vec 'Z a
  Vcons :: a -> Vec n a -> Vec ('S n) a

deriving instance (Show a) => Show (Vec n a)
