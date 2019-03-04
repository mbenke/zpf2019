---
title: Zaawansowane programowanie funkcyjne
author:  Marcin Benke
date: 12 marca 2019
---

<meta name="duration" content="80" />

# Testowanie programów w Haskellu
* doctest [github: sol/doctest](https://github.com/sol/doctest)
* HUnit
* Quickcheck
* QuickCheck + doctest
* Hedgehog [github: hedgehogqa/haskell-hedgehog](https://github.com/hedgehogqa/haskell-hedgehog)

# doctest

Przykłady w dokumetacji mogą służyć za testy regresji

``` {.haskell }
module DoctestExamples where
-- | Expect success
-- >>> 2 + 2
-- 4

-- | Expect failure
-- >>> 2 + 2
-- 5

```

```
$ stack install doctest
$ stack exec doctest DoctestExamples.hs
### Failure in DoctestExamples.hs:7: expression `2 + 2'
expected: 5
 but got: 4
Examples: 2  Tried: 2  Errors: 0  Failures: 1
```

# Przykład z BNFC
``` {.haskell}
-- | Generate a name in the given case style taking into account the reserved
-- word of the language.
-- >>> mkName [] SnakeCase "FooBAR"
-- "foo_bar"
-- >>> mkName [] CamelCase "FooBAR"
-- "FooBAR"
-- >>> mkName [] CamelCase "Foo_bar"
-- "FooBar"
-- >>> mkName ["foobar"] LowerCase "FooBAR"
-- "foobar_"
mkName :: [String] -> NameStyle -> String -> String
mkName reserved style s = ...
```

# Dygresja - Haddock

Haddock (<http://haskell.org/haddock>) jest standardowym narzedziem do dokuentowania kodu w Haskellu

Sekwencja "`-- |`" (spacja jest istotna) rozpoczyna blok komentarzy, który trafia do dokumentacji

``` haskell
-- |The 'square' function squares an integer.
-- It takes one argument, of type 'Int'.
square :: Int -> Int
square x = x * x
```

# HUnit

W większości języków powszechną praktyką jest stosowanie testów jednostkowych.

Mozna to robić i w Haskellu., np.

~~~~ {.haskell}
import Test.HUnit
import MyArray

main = runTestTT tests

tests = TestList [test1,test2]

listArray1 es = listArray (1,length es) es
test1 = TestCase$assertEqual "a!2 = 2" (listArray1 [1..3] ! 2) 2
test2 = TestCase$assertEqual "elems . array = id"
                             (elems $ listArray1 [1..3]) [1..3]
~~~~

albo

~~~~ {.haskell}
import Test.HUnit

run = runTestTT tests
tests = TestList [TestLabel "test1" test1, TestLabel "test2" test2]

test1 = TestCase (assertEqual "for (foo 3)," (1,2) (foo 3))
test2 = TestCase (do (x,y) <- partA 3
                     assertEqual "for the first result of partA," 5 x
                     b <- partB y
                     assertBool ("(partB " ++ show y ++ ") failed") b)
~~~~

~~~~
*Main Test.HUnit> run
Cases: 2  Tried: 2  Errors: 0  Failures: 0
Counts {cases = 2, tried = 2, errors = 0, failures = 0}

*Main Test.HUnit> :t runTestTT
runTestTT :: Test -> IO Counts
~~~~

# Posortujmy listę

~~~~ {.haskell}
mergeSort :: (a -> a -> Bool) -> [a] -> [a]
mergeSort pred = go
  where
    go []  = []
    go [x] = [x]
    go xs  = merge (go xs1) (go xs2)
      where (xs1,xs2) = split xs

    merge xs [] = xs
    merge [] ys = ys
    merge (x:xs) (y:ys)
      | pred x y  = x : merge xs (y:ys)
      | otherwise = y : merge (x:xs) ys
~~~~


# Funkcja split

...tworzy dwie podlisty podobnej długości, które będzie można po posortowaniu złączyć

~~~~ {.haskell}
split :: [a] -> ([a],[a])
split []       = ([],[])
split [x]      = ([x],[])
split (x:y:zs) = (x:xs,y:ys)
  where (xs,ys) = split zs
~~~~


# Sortowanie: testy jednostkowe


~~~~
sort = mergeSort ((<=) :: Int -> Int -> Bool)

sort [1,2,3,4] == [1,2,3,4]
sort [4,3,2,1] == [1,2,3,4]
sort [1,4,2,3] == [1,2,3,4]
...
~~~~

To się robi trochę nudne...

...ale dzięki typom, można lepiej.

# Własności

oczywista własność sortowania:

~~~~ {.haskell}
prop_idempotent = sort . sort == sort
~~~~

nie jest definiowalna; nie możemy porównywać funkcji.

Możemy "oszukać":

~~~~ {.haskell}
prop_idempotent xs =
    sort (sort xs) == sort xs
~~~~

Spróbujmy w interpreterze:

~~~~
*Main> prop_idempotent [3,2,1]
True
~~~~

# Próba mechanizacji

Możemy to próbować zmechanizować:

~~~~
prop_permute :: ([a] -> Bool) -> [a] -> Bool
prop_permute prop = all prop . permutations

*Main> prop_permute prop_idempotent [1,2,3]
True
*Main> prop_permute prop_idempotent [1..4]
True
*Main> prop_permute prop_idempotent [1..5]
True
*Main> prop_permute prop_idempotent [1..10]
  C-c C-cInterrupted.
~~~~

# QuickCheck

* Generowanie dużej ilości testów jednostkowych jest żmudne

* Sprawdzenie wszystkich możliwości jest nierealistyczne

* Pomysł: wygenerować odpowiednią losową próbkę danych

~~~~
*Main> import Test.QuickCheck
*Main Test.QuickCheck> quickCheck prop_idempotent
+++ OK, passed 100 tests.
~~~~

QuickCheck wylosował 100 list i sprawdził własność,

Możemy zażyczyć sobie np. 1000:

~~~~
*Main Test.QuickCheck> quickCheckWith stdArgs {maxSuccess = 1000}  prop_idempotent
+++ OK, passed 1000 tests.
~~~~

Uwaga: nie możemy losować  wartości polimorficznych, dlatego `prop_idempotent` monomorficzne.

**Ćwiczenie:** napisz i uruchom kilka testów dla sortowania i kilka
  testów dla własnych funkcji.

# Jak to działa?

Dla uproszczenia najpierw przyjrzyjmy się starszej wersji

Główne składniki

~~~~ {.haskell}
quickCheck  :: Testable a => a -> IO ()
quickCheck   = check quick


check :: Testable a => Config -> a -> IO ()
quick :: Config

class Testable a where
  property :: a -> Property

instance Testable Bool where...

instance (Arbitrary a, Show a, Testable b) => Testable (a -> b) where
  property f = forAll arbitrary f

class Arbitrary a where
  arbitrary   :: Gen a

instance Monad Gen where ...
~~~~

# Generacja liczb losowych

~~~~ {.haskell}
import System.Random
  ( StdGen       -- :: *
  , newStdGen    -- :: IO StdGen
  , randomR      -- :: (RandomGen g, Random a) => (a, a) -> g -> (a, g)
  , split        -- :: RandomGen g => g -> (g, g)
                 -- rozdziela argument na dwa niezależne generatory
  -- instance RandomGen StdGen
  -- instance Random Int
  )

roll :: StdGen -> Int
roll rnd = fst $ randomR (1,6) rnd
main = do
  rnd <- newStdGen
  let (r1,r2) = split rnd
  print (roll r1)
  print (roll r2)
  print (roll r1)
  print (roll r2)
~~~~

~~~~
*Main System.Random> main
4
5
4
5
~~~~


# Generatory losowych obiektów


~~~~ {.haskell}
choose :: (Int,Int) -> Gen Int
oneof :: [Gen a] -> Gen a

instance Arbitrary Int where
    arbitrary = choose (-100, 100)

data Colour = Red | Green | Blue
instance Arbitrary Colour where
    arbitrary = oneof [return Red, return Green, return Blue]

instance Arbitrary a => Arbitrary [a] where
    arbitrary = oneof [return [], liftM2 (:) arbitrary arbitrary]

generate :: Gen a -> IO a
~~~~

Jaka jest oczekiwana długość wylosowanej listy?

$$ \sum_{n=0}^\infty {n\over 2^{n+1}} = 1 $$

# Dopasowanie rozkładu

~~~~ {.haskell}
frequency :: [(Int, Gen a)] -> Gen a
instance Arbitrary a => Arbitrary [a] where
    arbitrary = frequency
        [ (1, return [])
	    , (4, liftM2 (:) arbitrary arbitrary])
		]

data Tree a = Leaf a | Branch (Tree a) (Tree a)
instance Arbitrary a => Arbitrary (Tree a) where
    arbitrary = frequency
        [(1, liftM Leaf arbitrary)
        ,(2, liftM2 Branch arbitrary arbitrary)
        ]

threetrees :: Gen [Tree Int]
threetrees = sequence [arbitrary, arbitrary, arbitrary]
~~~~

jakie jest prawdopodobieństwo że generowanie 3 drzew się zatrzyma?

<!---

Dla jednego drzewa:

$$ p = {1\over 3} + {2\over 3} p^2 $$

$$ p = 1/2 $$

-->

# Ograniczanie rozmiaru

~~~~ {.haskell}
-- generator bierze pożądany rozmiar i StdGen i daje a
newtype Gen a = Gen (Int -> StdGen -> a)

chooseInt1 :: (Int,Int) -> Gen Int
chooseInt1 bounds = Gen $ \n r  -> fst (randomR bounds r)

-- | `sized` tworzy generator z rodziny generatorów indeksowanej rozmiarem
sized :: (Int -> Gen a) -> Gen a
sized fgen = Gen (\n r -> let Gen m = fgen n in m n r)

-- | `resize` tworzy generator stałego rozmiaru
resize :: Int -> Gen a -> Gen a
resize n (Gen m) = Gen (\_ r -> m n r)
~~~~

# Lepsze Arbitrary dla Tree

~~~~ {.haskell}
instance Arbitrary a => Arbitrary (Tree a) where
    arbitrary = sized arbTree

arbTree 0 = liftM Leaf arbitrary
arbTree n = frequency
        [(1, liftM Leaf arbitrary)
        ,(4, liftM2 Branch (arbTree (div n 2))(arbTree (div n 2)))
        ]
~~~~

# Monada generatorów

~~~~ {.haskell}
-- Trochę jak monada stanu, tylko musimy rozdzielić "stan" na dwa
instance Monad Gen where
  return a = Gen $ \n r -> a
  Gen m >>= k = Gen $ \n r0 ->
    let (r1,r2) = split r0
        Gen m'  = k (m n r1)
     in m' n r2

instance Functor Gen where
  fmap f m = m >>= return . f

chooseInt :: (Int,Int) -> Gen Int
chooseInt bounds = (fst . randomR bounds) `fmap` rand

rand :: Gen StdGen
rand = Gen (\n r -> r)

choose ::  Random a => (a, a) -> Gen a
choose bounds = (fst . randomR bounds) `fmap` rand
~~~~

# Arbitrary

~~~~ {.haskell}
class Arbitrary a where
  arbitrary   :: Gen a

elements :: [a] -> Gen a
elements xs = (xs !!) `fmap` choose (0, length xs - 1)

vector :: Arbitrary a => Int -> Gen [a]
vector n = sequence [ arbitrary | i <- [1..n] ]
-- sequence :: Monad m => [m a] -> m [a]
instance Arbitrary () where
  arbitrary = return ()

instance Arbitrary Bool where
  arbitrary     = elements [True, False]

instance Arbitrary a => Arbitrary [a] where
  arbitrary          = sized (\n -> choose (0,n) >>= vector)

instance Arbitrary Int where
  arbitrary     = sized $ \n -> choose (-n,n)
~~~~

# Result - wynik testu

Test może dać trojaki wynik:

* Just True - sukces
* Just False - porażka  (plus kontrprzykład)
* Nothing - dane nie nadawały się do testu

~~~~ {.haskell}
data Result = Result { ok :: Maybe Bool, arguments :: [String] }

nothing :: Result
nothing = Result{ ok = Nothing,  arguments = [] }

newtype Property
  = Prop (Gen Result)
~~~~

Własność (`Property`),  to obliczenie w monadzie `Gen` dające `Result`

# Testable

Aby coś przetestować musimy mieć dla tego generator wyników:

~~~~ {.haskell}
class Testable a where
  property :: a -> Property

result :: Result -> Property
result res = Prop (return res)

instance Testable () where
  property () = result nothing

instance Testable Bool where
  property b = result (nothing { ok = Just b })

instance Testable Property where
  property prop = prop
~~~~

~~~~
*SimpleCheck1> check True
OK, passed 100 tests
*SimpleCheck1> check False
Falsifiable, after 0 tests:
~~~~

# Uruchamianie testów

~~~~ {.haskell}
generate :: Int -> StdGen -> Gen a -> a

tests :: Gen Result -> StdGen -> Int -> Int -> IO ()
tests gen rnd0 ntest nfail
  | ntest == configMaxTest = do done "OK, passed" ntest
  | nfail == configMaxFail = do done "Arguments exhausted after" ntest
  | otherwise               =
         case ok result of
           Nothing    ->
             tests gen rnd1 ntest (nfail+1)
           Just True  ->
             tests gen rnd1 (ntest+1) nfail
           Just False ->
             putStr ( "Falsifiable, after "
                   ++ show ntest
                   ++ " tests:\n"
                   ++ unlines (arguments result)
                    )
     where
      result      = generate (configSize ntest) rnd2 gen
      (rnd1,rnd2) = split rnd0
~~~~


# forAll

~~~~ {.haskell}
-- | `evaluate` oblicza generator z instancji Testable
evaluate :: Testable a => a -> Gen Result
evaluate a = gen where Prop gen = property a

forAll :: (Show a, Testable b) => Gen a -> (a -> b) -> Property
forAll gen body = Prop $
  do a   <- gen
     res <- evaluate (body a)
     return (argument a res)
 where
  argument a res = res{ arguments = show a : arguments res }


propAddCom1, propAddCom2 :: Property
propAddCom1 =  forAll (chooseInt (-100,100)) (\x -> x + 1 == 1 + x)
propAddCom2 =  forAll int (\x -> forAll int (\y -> x + y == y + x)) where
  int = chooseInt (-100,100)
~~~~

~~~~
>>> check $ forAll (chooseInt (-100,100)) (\x -> x + 0 == x)
OK, passed 100 tests
>>> check $ forAll (chooseInt (-100,100)) (\x -> x + 1 == x)
Falsifiable, after 0 tests:
-22
~~~~

# Funkcje i implikacja

Mając `forAll`, funkcje są zaskakująco łatwe:

~~~~ {.haskell}
instance (Arbitrary a, Show a, Testable b) => Testable (a -> b) where
  property f = forAll arbitrary f

propAddCom3 :: Int -> Int -> Bool
propAddCom3 x y = x + y == y + x
~~~~

Jeszcze implikacja: jeśli p to q

~~~~ {.haskell}
(==>) :: Testable a => Bool -> a -> Property
True  ==> a = property a
False ==> a = property () -- nieadekwatne dane

propMul1 :: Int -> Property
propMul1 x = (x>0) ==> (2*x > 0)

propMul2 :: Int -> Int -> Property
propMul2 x y = (x>0) ==> (x*y > 0)
~~~~

~~~~
> check propMul1
OK, passed 100 tests

> check propMul2
Falsifiable, after 0 tests:
2
-2
~~~~


# Generowanie funkcji

Potrafimy testować funkcje, ale czy potrafimy wygenerować losową funkcję?

Zauważmy, że

~~~~ {.haskell}
Gen a ~ (Int -> StdGen -> a)
Gen(a -> b) ~ (Int -> StdGen -> a -> b) ~ (a -> Gen b)
~~~~

możemy więc napisać funkcję

~~~~ {.haskell}
promote :: (a -> Gen b) -> Gen (a -> b)
promote f = Gen (\n r -> \a -> let Gen m = f a in m n r)
~~~~

Możemy uzyć `promote` do skonstruowania generatora dla funkcji, jeśli tylko potrafimy skonstruować generator dla wyników zależący jakoś od argumentów.

# Coarbitrary

Możemy to opisać klasą:

~~~~ {.haskell}
class CoArbitrary a where
  coarbitrary :: a -> Gen b -> Gen b
~~~~

Na podstawie wartości argumentu, `coarbitrary` tworzy transformator generatorów.

Teraz możemy użyć `Coarbitrary` by stworzyć `Arbitrary` dla funkcji:

~~~~ {.haskell}
instance (CoArbitrary a, Arbitrary b) => Arbitrary(a->b) where
  arbitrary = promote $ \a -> coarbitrary a arbitrary
~~~~

NB w rzeczywistości w QuickChecku `coarbitrary` jest metodą klasy `Arbitrary`.

**Ćwiczenie:** napisz kilka instancji `Arbitrary` dla swoich typów. Możesz zacząć od `coarbitrary = undefined`

# Instancje CoArbitrary

Żeby definiować instancje CoArbitrary

~~~~ {.haskell}
class CoArbitrary where
  coarbitrary :: a -> Gen b -> Gen b
~~~~

musimy umieć pisać transformatory generatorów. Zdefiniujmy funkcję

~~~~ {.haskell}
variant :: Int -> Gen a -> Gen a
variant v (Gen m) = Gen (\n r -> m n (rands r !! (v+1)))
 where
  rands r0 = r1 : rands r2 where (r1, r2) = split r0
~~~~

która rozdziela generator liczb losowych na odpowiednią ilość i
wybiera jeden z nich zależnie od wartości argumentu.

~~~~ {.haskell}
instance CoArbitrary Bool where
  coarbitrary False = variant 0
  coarbitrary True  = variant 1
~~~~

# Własności funkcji

~~~~ {.haskell}
infix 4 ===
(===)  f g x = f x == g x

instance Show(a->b) where
  show f = "<function>"

propCompAssoc f g h = (f . g) . h === f . (g . h)
  where types = [f,g,h::Int->Int]
~~~~

# Problem z implikacją

~~~~
prop_insert1 x xs = ordered (insert x xs)

*Main Test.QuickCheck> quickCheck prop_insert1
*** Failed! Falsifiable (after 6 tests and 7 shrinks):
0
[0,-1]
~~~~

...oczywiście...

~~~~
prop_insert2 x xs = ordered xs ==> ordered (insert x xs)

>>> quickCheck prop_insert2
*** Gave up! Passed only 43 tests.
~~~~

Prawdopodobieństwo, że losowa lista jest posortowana jest niewielkie :)

~~~~
prop_insert3 x xs = collect (length xs) $  ordered xs ==> ordered (insert x xs)

>>> quickCheck prop_insert3
*** Gave up! Passed only 37 tests:
51% 0
32% 1
16% 2
~~~~

...a i te posortowane są mało przydatne.

# Czasami trzeba napisać własny generator

* Trzeba zdefiniować nowy typ (chyba, że już mamy)

~~~~
newtype OrderedInts = OrderedInts [Int]

prop_insert4 :: Int -> OrderedInts -> Bool
prop_insert4  x (OrderedInts xs) = ordered (insert x xs)

>>> sample (arbitrary:: Gen OrderedInts)
OrderedInts []
OrderedInts [0,0]
OrderedInts [-2,-1,2]
OrderedInts [-4,-2,0,0,2,4]
OrderedInts [-7,-6,-6,-5,-2,-1,5]
OrderedInts [-13,-12,-11,-10,-10,-7,1,1,1,10]
OrderedInts [-13,-10,-7,-5,-2,3,10,10,13]
OrderedInts [-19,-4,26]
OrderedInts [-63,-15,37]
OrderedInts [-122,-53,-47,-43,-21,-19,29,53]
~~~~

# doctest + QuickCheck

~~~~ {.haskell}
module Fib where

-- $setup
-- >>> import Control.Applicative
-- >>> import Test.QuickCheck
-- >>> newtype Small = Small Int deriving Show
-- >>> instance Arbitrary Small where arbitrary = Small . (`mod` 10) <$> arbitrary

-- | Compute Fibonacci numbers
--
-- The following property holds:
--
-- prop> \(Small n) -> fib n == fib (n + 2) - fib (n + 1)
fib :: Int -> Int
fib 0 = 0
fib 1 = 1
fib n = fib (n - 1) + fib (n - 2)
~~~~

```
stack install QuickCheck
stack exec doctest Fib.hs
Run from outside a project, using implicit global project config
Using resolver: lts-9.21 from implicit global project's config file: /Users/ben/.stack/global/stack.yaml
Examples: 5  Tried: 5  Errors: 0  Failures: 0
```

# Uruchamianie wszystkich testów w module

Funkcja `quickCheckAll` pozwala na przetestowanie wszystkich własności o nazwach zaczynających się od `prop_`. Wykorzystuje do tego TemplateHaskell.

Na kolejnym wykładzie dowiemy się jak działają takie funkcje.

Przykład użycia

``` haskell
{-# LANGUAGE TemplateHaskell #-}
import Test.QuickCheck

prop_AddCom3 :: Int -> Int -> Bool
prop_AddCom3 x y = x + y == y + x

prop_Mul1 :: Int -> Property
prop_Mul1 x = (x>0) ==> (2*x > 0)

return []
runTests = $quickCheckAll

main = runTests
```
