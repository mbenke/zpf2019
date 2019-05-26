{-# LANGUAGE GADTs #-}
{-# LANGUAGE DataKinds, KindSignatures #-}
{-# LANGUAGE TypeFamilies, TypeOperators #-}
{-# LANGUAGE StandaloneDeriving #-}
{-# LANGUAGE UndecidableInstances #-}

module Promo2 where
import qualified GHC.TypeLits as Lit


type family Lit n where
    Lit 0 = Z
    Lit n = S (Lit (n Lit.- 1))

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


infixl 6 :+

type family (n :: Nat) :+ (m :: Nat) :: Nat
type instance Z :+ m = m
type instance (S n) :+ m = S (n :+ m)

vhead :: Vec (S n) a -> a
vhead (Vcons x _) = x

vtail :: Vec (S n) a -> Vec n a
vtail (Vcons _ xs) = xs

vapp :: Vec m a -> Vec n a -> Vec (m :+ n) a
vapp Vnil ys = ys
vapp (Vcons x xs) ys = Vcons x (vapp xs ys)

-- Indexing
-- atIndex :: Vec n a -> (m < n) -> a

data Fin n where
    FinZ :: Fin (S n) -- zero is less than any successor
    FinS :: Fin n -> Fin (S n) -- n is less than (n+1)


atIndex :: Vec n a -> Fin n -> a
atIndex (Vcons x xs) FinZ = x
atIndex (Vcons x xs) (FinS k) = atIndex xs k
-- Question - why not:
-- atIndex :: Vec (S n) a -> ... ?

-- Want
-- replicate :: Nat -> a -> Vec n a
-- replicate Z _ = Vnil -- doesn't work

-- inhabitants of Nat types
data SNat n where
  SZ :: SNat Z
  SS :: SNat n -> SNat (S n)
deriving instance Show(SNat n)

vreplicate :: SNat n -> a -> Vec n a
vreplicate SZ _ = Vnil
vreplicate (SS n) x = Vcons x (vreplicate n x)

add :: (SNat m) -> (SNat n) -> SNat(m :+ n)
add SZ n = n
add (SS m) n = SS (add m n)

-- Exercise: define multiplication

-- Other promotions
data HList :: [*] -> * where
  HNil  :: HList '[]
  HCons :: a -> HList t -> HList (a ': t)

data Tuple :: (*,*) -> * where
  Tuple :: a -> b -> Tuple '(a,b)

foo0 :: HList '[]
foo0 = HNil

foo1 :: HList '[Int]
foo1 = HCons (3::Int) HNil

foo2 :: HList [Int, Bool]
foo2 = undefined  -- (easy) exercise
