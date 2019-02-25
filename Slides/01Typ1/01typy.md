% Zaawansowane programowanie funkcyjne
% Marcin Benke
% 26 lutego 2019

<meta name="duration" content="80" />

# Plan wykładu
* Typy i klasy
    * Typy algebraiczne i klasy typów
    * Klasy konstruktorowe
    * Klasy wieloparametrowe, zależności funkcyjne
* Testowanie (QuickCheck)
* Typy zależne, Agda, Idris, Coq, dowodzenie własności (ok. 7 wykladów)
* Typy zależne w Haskellu
    * Rodziny typów, typy skojarzone, uogólnione typy algebraiczne   (GADT)
    * data kinds, kind polymorphism
* Metaprogramowanie
* Programowanie równoległe i współbieżne w Haskellu
    * Programowanie wielordzeniowe i wieloprocesorowe (SMP)
    * Równoległość danych (Data Parallel Haskell)
* Prezentacje projektów

Jakieś życzenia?

# Zasady zaliczania
* Laboratorium: zdefiniowane zadanie Coq + **prosty** projekt 1-3 osobowy - Haskell
* Egzamin ustny, którego istotną częścią jest prezentacja projektu.
* Alternatywna forma zaliczenia: referat (koniecznie ciekawy!)
* ...możliwe  także inne formy.

# Materiały

~~~~~
$ cabal update
$ cabal install pandoc
$ PATH=~/.cabal/bin:$PATH            # Linux
$ PATH=~/Library/Haskell/bin:$PATH   # OS X
$ git clone git://github.com/mbenke/zpf2018.git
$ cd zpf2018/Slides
$ make
~~~~~

albo, przy użyciu stack - https://haskellstack.org/

~~~~
stack setup
stack install pandoc
export PATH=$(stack path --local-bin):$PATH
...
~~~~

Na students można jak wyżej albo jeśli brakuje quota, z użyciem systemowego GHC:

~~~~
export PATH=/home/students/inf/PUBLIC/MRJP/ghc-8.2.2/bin:$PATH
export STACK="/home/students/inf/PUBLIC/MRJP/Stack/stack --system-ghc --resolver ghc-8.2"
$STACK setup
$STACK config set system-ghc true
$STACK  upgrade --force-download  # or cp stack executable to your path
#  ...
#  Should I try to perform the file copy using sudo? This may fail
#  Try using sudo? (y/n) n

export PATH=$($STACK path --local-bin):$PATH
stack install pandoc
~~~~

# Dygresja - cabal i stack

**Common Architecture for Building Applications and Libraries**

`cabal install` -  pozwala instalować biblioteki na swoim koncie, bez uprawnień administratora

```
[ben@students Haskell]$ cabal update
Downloading the latest package list
  from hackage.haskell.org
[ben@students Haskell]$ cabal install GLFW
...kompilacja...
Installing library in
 /home/staff/iinf/ben/.cabal/lib/GLFW-0.4.2/ghc-6.10.4
Registering GLFW-0.4.2...
Reading package info from "dist/installed-pkg-config"
 ... done.
Writing new package config file... done.
```

Wiele bibliotek na `http://hackage.haskell.org/`

# Cabal hell

```
$ cabal install criterion
Resolving dependencies...
In order, the following would be installed:
monad-par-extras-0.3.3 (reinstall) changes: mtl-2.1.2 -> 2.2.1,
transformers-0.3.0.0 -> 0.5.2.0
nats-1.1.1 (reinstall) changes: hashable-1.1.2.5 -> 1.2.5.0
...
criterion-1.1.4.0 (new package)
cabal: The following packages are likely to be broken by the reinstalls:
monad-par-0.3.4.7
void-0.7.1
lens-4.15.1
...
HTTP-4000.3.3
Use --force-reinstalls if you want to install anyway.
```

W nowszych wersjach cabal częściowo rozwiązane przez sandboxing i `cabal new-install`
# Stack + stackage

> Stackage is a stable source of Haskell packages. We guarantee that packages build consistently and pass tests before generating nightly and Long Term Support (LTS) releases.

```
LTS 13.9 for GHC 8.6.3, published a day ago
LTS 12.26 for GHC 8.4.4, published a month ago
LTS 12.14 for GHC 8.4.3, published 4 months ago
LTS 11.22 for GHC 8.2.2, published 6 months ago
LTS 9.21 for GHC 8.0.2, published a year ago
LTS 7.24 for GHC 8.0.1, published a year ago
LTS 6.35 for GHC 7.10.3, published a year ago
LTS 3.22 for GHC 7.10.2, published 3 years ago
LTS 2.22 for GHC 7.8.4, published 4 years ago
LTS 0.7 for GHC 7.8.3, published 4 years ago
```

```
$ stack --resolver lts-3.22 install criterion
Run from outside a project, using implicit global project config
Using resolver: lts-3.22 specified on command line
Downloaded lts-3.22 build plan.
mtl-2.2.1: using precompiled package
...
criterion-1.1.0.0: download
criterion-1.1.0.0: configure
criterion-1.1.0.0: build
criterion-1.1.0.0: copy/register

```

# Budowanie projektu

```
$ stack new hello --resolver lts-11.22 && cd hello
Downloading template "new-template" to create project "hello" in hello/ ...

Selected resolver: lts-11.22
Initialising configuration using resolver: lts-11.22
Total number of user packages considered: 1
Writing configuration to file: hello/stack.yaml
All done.

$ stack build
Building all executables for `hello' once. After a successful build of all of th
em, only specified executables will be rebuilt.
hello-0.1.0.0: configure (lib + exe)
...
hello-0.1.0.0: copy/register
Installing library in /home/staff/iinf/ben/tmp/hello/.stack-work/install/x86_64-linux/lts-11.22/8.2.2/lib/x86_64-linux-ghc-8.2.2/hello-0.1.0.0-CaHXYhIIKYt3q9LDFmJN3m
Installing executable hello-exe in /home/staff/iinf/ben/tmp/hello/.stack-work/install/x86_64-linux/lts-11.22/8.2.2/bin
Registering library for hello-0.1.0.0..
$ stack exec hello-exe
someFunc
```

# Stack - ćwiczenia

1.  Na własnym komputerze
    * Zainstalować `stack`
    * Zainstalować GHC 7.10 przy pomocy `stack setup`
    * Zainstalować GHC 8 przy pomocy `stack setup`
    * Uruchomić `stack ghci` 7.10 i 8
    * Zbudować i uruchomić projekt hello, trochę go zmodyfikować

2. Na students
    * Można zrobić to samo co powyżej, ale możliwy problem z quota
    * `stack setup` przy użyciu GHC 8.2 (i może 7.10) z PUBLIC
    * `stack config set system-ghc --global true`
    * Reszta jak wyżej

# Języki funkcyjne
* typowane dynamicznie, gorliwe: Lisp
* typowane statycznie, gorliwe, nieczyste: ML
* typowane statycznie, leniwe, czyste: Haskell

Ten wykład: Haskell, ze szczególnym naciskiem na typy.

Bogata struktura typów jest tym, co wyróżnia Haskell wśród innych języków.

# Typy jako język specyfikacji

Typ funkcji często specyfikuje nie tylko jej wejście i wyjście ale i relacje między nimi:

~~~~ {.haskell}
f :: forall a. a -> a
f x = ?
~~~~

Jeśli `(f x)` daje wynik, to musi nim być `x`

* Philip Wadler "Theorems for Free"

* Funkcja typu `a -> IO b` może mieć efekty uboczne

    ~~~~ {.haskell}
    import Data.IORef

    f :: Int -> IO (IORef Int)
    f i = do
      print i
      r <- newIORef i
      return r

    main = do
      r <- f 42
      j <- readIORef r
      print j
    ~~~~



# Typy jako język specyfikacji (2)

Funkcja typu `Integer -> Integer` zasadniczo nie może mieć efektów ubocznych

Liczby Fibonacciego w stałej pamięci

~~~~ {.haskell}
import Control.Monad.ST
import Data.STRef
fibST :: Integer -> Integer
fibST n =
    if n < 2 then n else runST fib2 where
      fib2 =  do
        x <- newSTRef 0
        y <- newSTRef 1
        fib3 n x y

      fib3 0 x _ = readSTRef x
      fib3 n x y = do
              x' <- readSTRef x
              y' <- readSTRef y
              writeSTRef x y'
              writeSTRef y (x'+y')
              fib3 (n-1) x y
~~~~

Jak to?

~~~~
runST :: (forall s. ST s a) -> a
~~~~

Typ `runST` gwarantuje, że efekty uboczne nie wyciekają. Funkcja `fibST`
jest czysta.

# Typy jako język projektowania

* Projektowanie programu przy użyciu typów i `undefined`

    ~~~~ {.haskell}
    conquer :: [Foo] -> [Bar]
    conquer fs = concatMap step fs

    step :: Foo -> [Bar]
    step = undefined
    ~~~~

# Typy jako język programowania

*    Funkcje na typach obliczane w czasie kompilacji

    ~~~~ {.haskell}
    data Zero
    data Succ n

    type One   = Succ Zero
    type Two   = Succ One
    type Three = Succ Two
    type Four  = Succ Three

    one   = undefined :: One
    two   = undefined :: Two
    three = undefined :: Three
    four  = undefined :: Four

    class Add a b c | a b -> c where
      add :: a -> b -> c
      add = undefined
    instance              Add  Zero    b  b
    instance Add a b c => Add (Succ a) b (Succ c)
    ~~~~

    ~~~~
    *Main> :t add three one
    add three one :: Succ (Succ (Succ (Succ Zero)))
    ~~~~

* Ćwiczenie: rozszerzyć o mnożenie i silnię

# Typy jako język programowania (2)
Wektory przy użyciu klas:

~~~~ {.haskell}
data Vec :: * -> * -> * where
  VNil :: Vec Zero a
  (:>) :: a -> Vec n a -> Vec (Succ n) a

vhead :: Vec (Succ n) a -> a
vhead (x :> xs) = x
~~~~

**Ćwiczenie:** dopisać `vtail`, `vlast`

Chcielibyśmy również mieć

~~~~ {.haskell}
vappend :: Add m n s => Vec m a -> Vec n a -> Vec s a
~~~~

ale tu niestety podstawowy system typów okazuje się za słaby - więcej wkrótce

# Typy jako język programowania (3)

* Wektory przy użyciu rodzin typów:

    ~~~~ {.haskell}
    data Zero = Zero
    data Suc n = Suc n

    type family m :+ n
    type instance Zero :+ n = n
    type instance (Suc m) :+ n = Suc(m:+n)

    data Vec :: * -> * -> * where
      VNil :: Vec Zero a
      (:>) :: a -> Vec n a -> Vec (Suc n) a

    vhead :: Vec (Suc n) a -> a
    vappend :: Vec m a -> Vec n a -> Vec (m:+n) a
    ~~~~


# Typy zależne

Prawdziwe programowanie na poziomie typów  i dowodzenie własności programów możliwe w języku z typami zależnymi, takim jak Agda, Epigram, Idris

~~~~
module Data.Vec where
infixr 5 _∷_

data Vec (A : Set a) : ℕ → Set where
  []  : Vec A zero
  _∷_ : ∀ {n} (x : A) (xs : Vec A n) → Vec A (suc n)

_++_ : ∀ {a m n} {A : Set a} → Vec A m → Vec A n → Vec A (m + n)
[]       ++ ys = ys
(x ∷ xs) ++ ys = x ∷ (xs ++ ys)

module UsingVectorEquality {s₁ s₂} (S : Setoid s₁ s₂) where
  xs++[]=xs : ∀ {n} (xs : Vec A n) → xs ++ [] ≈ xs
  xs++[]=xs []       = []-cong
  xs++[]=xs (x ∷ xs) = SS.refl ∷-cong xs++[]=xs xs
~~~~


# Problem z typami zależnymi

O ile Haskell bywa czasami nieczytelny, to z typami zależnymi całkiem łatwo przesadzić:

~~~~
  now-or-never : Reflexive _∼_ →
                 ∀ {k} (x : A ⊥) →
                 ¬ ¬ ((∃ λ y → x ⇓[ other k ] y) ⊎ x ⇑[ other k ])
  now-or-never refl x = helper <$> excluded-middle
    where
    open RawMonad ¬¬-Monad

    not-now-is-never : (x : A ⊥) → (∄ λ y → x ≳ now y) → x ≳ never
    not-now-is-never (now x)   hyp with hyp (, now refl)
    ... | ()
    not-now-is-never (later x) hyp =
      later (♯ not-now-is-never (♭ x) (hyp ∘ Prod.map id laterˡ))

    helper : Dec (∃ λ y → x ≳ now y) → _
    helper (yes ≳now) = inj₁ $ Prod.map id ≳⇒ ≳now
    helper (no  ≵now) = inj₂ $ ≳⇒ $ not-now-is-never x ≵now
~~~~

...chociaż oczywiście pisanie takich dowodów jest świetną zabawą.

# Parallel Haskell

Równoległe rozwiązywanie Sudoku

~~~~ {.haskell}
main = do
    [f] <- getArgs
    grids <- fmap lines $ readFile f
    runEval (parMap solve grids) `deepseq` return ()

parMap :: (a -> b) -> [a] -> Eval [b]
parMap f [] = return []
parMap f (a:as) = do
   b <- rpar (f a)
   bs <- parMap f as
   return (b:bs)

solve :: String -> Maybe Grid
~~~~

~~~~
$ ./sudoku3b sudoku17.1000.txt +RTS -N2 -s -RTS
  TASKS: 4 (1 bound, 3 peak workers (3 total), using -N2)
  SPARKS: 1000 (1000 converted, 0 overflowed, 0 dud, 0 GC'd, 0 fizzled)

  Total   time    2.84s  (  1.49s elapsed)
  Productivity  88.9% of total user, 169.6% of total elapsed

-N8: Productivity  78.5% of total user, 569.3% of total elapsed
N16: Productivity  62.8% of total user, 833.8% of total elapsed
N32: Productivity  43.5% of total user, 1112.6% of total elapsed
~~~~

# Parallel Fibonacci

~~~~ {.haskell}
cutoff :: Int
cutoff = 20

parFib n | n < cutoff = fib n
parFib n = p `par` q `pseq` (p + q)
    where
      p = parFib $ n - 1
      q = parFib $ n - 2

fib n | n<2 = n
fib n = fib (n - 1) + fib (n - 2)
~~~~

~~~~
./parfib +RTS -N60 -s -RTS
 SPARKS: 118393 (42619 converted, 0 overflowed, 0 dud,
                 11241 GC'd, 64533 fizzled)

  Total   time   17.91s  (  0.33s elapsed)
  Productivity  98.5% of total user, 5291.5% of total elapsed

-N60, cutoff=15
  SPARKS: 974244 (164888 converted, 0 overflowed, 0 dud,
                  156448 GC'd, 652908 fizzled)
  Total   time   13.59s  (  0.28s elapsed)
  Productivity  97.6% of total user, 4746.9% of total elapsed
~~~~

# Data Parallel Haskell

Dokąd chcemy dojść:

~~~~ {.haskell}
{-# LANGUAGE ParallelArrays #-}
{-# OPTIONS_GHC -fvectorise #-}

module DotP where
import qualified Prelude
import Data.Array.Parallel
import Data.Array.Parallel.Prelude
import Data.Array.Parallel.Prelude.Double as D

dotp_double :: [:Double:] -> [:Double:] -> Double
dotp_double xs ys = D.sumP [:x * y | x <- xs | y <- ys:]
~~~~

Wygląda jak operacja na listach, ale działa na tablicach i
"automagicznie" zrównolegla się na dowolną liczbę rdzeni/procesorów
(także CUDA).

Po drodze czeka nas jednak trochę pracy.

# Typy w Haskellu

* typy bazowe: `zeroInt :: Int`
* typy funkcyjne: `plusInt :: Int -> Int -> Int`
* typy polimorficzne `id :: a -> a`

    ~~~~ {.haskell}
    {-# LANGUAGE ExplicitForAll #-}
    g :: forall b.b -> b
    ~~~~

* typy algebraiczne

    ~~~~ {.haskell}
    data Tree a = Leaf | Node a (Tree a) (Tree a)
    ~~~~

* `Leaf` i `Node` są konstruktorami wartości:

    ~~~~ {.haskell}
    data Tree a where
    	 Leaf :: Tree a
         Node :: a -> Tree a -> Tree a -> Tree a
    ~~~~

* `Tree` jest *konstruktorem typowym*, czyli operacją na typach

* NB od niedawna Haskell dopuszcza puste typy:

    ~~~~ {.haskell}
    data Zero
    ~~~~

# Typowanie polimorficzne

* Generalizacja:

$${\Gamma \vdash e :: t, a \notin FV( \Gamma )}\over {\Gamma \vdash e :: \forall a.t}$$

 <!--
Jeśli $\Gamma \vdash e :: t, a \notin FV( \Gamma )$

to $\Gamma \vdash e :: \forall a.t$

  Γ ⊢ e :: t, a∉FV(Γ)
$$\Gamma \vdash e :: t$$ ,
 \(a \not\in FV(\Gamma) \) ,
to $\Gamma \vdash e :: \forall a.t$
-->

Na przykład

$${ { \vdash map :: (a\to b) \to [a] \to [b] } \over
   { \vdash map :: \forall b. (a\to b) \to [a] \to [b] } } \over
   { \vdash map :: \forall a. \forall b. (a\to b) \to [a] \to [b] } $$

Uwaga:

$$ f : a \to b \not \vdash map\; f :: \forall b. [a] \to [b]  $$

* Instancjacja

$$ {\Gamma \vdash e :: \forall a.t}\over {\Gamma \vdash e :: t[a:=s]} $$

# Klasy

* klasy opisują własności typów

    ~~~~ {.haskell}
    class Eq a where
      (==) :: a -> a -> Bool
    instance Eq Bool where
       True  == True  = True
       False == False = True
       _     == _     = False

    class Eq a => Ord a where ...
    ~~~~

* funkcje mogą być definiowane w kontekście klas:

    ~~~~ {.haskell}
    elem :: Eq a => a -> [a] -> Bool
    ~~~~

+ Implementacja
    - instancja tłumaczona na słownik metod (coś \'a la  vtable w C++)
    - kontekst (np Eq a) jest tłumaczony na ukryty parametr (słownik metod )
    - podklasa tłumaczona na funkcję


# Operacje na typach

* Prosty przykład:

    ~~~~ {.haskell}
    data Tree a = Leaf | Node a (Tree a) (Tree a)
    ~~~~

* Konstruktory typowe transformują typy

* `Tree` może zamienić np. `Int` w drzewo

+ Funkcje wyższego rzędu transformują funkcje

+ Konstruktory wyższego rzędu transformują konstruktory typów

~~~~ {.haskell}
newtype IdentityT m a = IdentityT { runIdentityT :: m a }
~~~~

# Klasy konstruktorowe

* klasy konstruktorowe opisują własności konstruktorów typów:

    ~~~~ {.haskell}
    class Functor f where
      fmap :: (a->b) -> f a -> f b
    (<$>) = fmap

    instance Functor [] where
      fmap = map

    class Functor f => Pointed f where
       pure :: a -> f a
    instance Pointed [] where
       pure = (:[])

    class Pointed f => Applicative f where
      (<*>) :: f(a->b) -> f a -> f b

    instance Applicative [] where
      fs <*> xs = concat $ flip map fs (flip map xs)

    class Applicative m => Monad' m where
      (>>=) :: m a -> (a -> m b) -> m b
    ~~~~

<!--

    class Pointed f => Applicative f where
      (<*>) :: f(a->b) -> f a -> f b
      (*>) :: f a -> f b -> f b
      x *> y = (flip const) <$> x <*> y
      (<*) :: f a -> f b -> f a
      x <* y = const <$> x <*> y

    liftA2 :: Applicative f => (a -> b -> c) -> f a -> f b -> f c
    liftA2 f a b = f <$> a <*> b

-->

# Rodzaje (kinds)

* Operacje na wartościach są opisywane przez ich typy

* Operacje na typach są opisywane przez ich rodzaje (kinds)

* Typy (np. `Int`) są rodzaju `*`

* Jednoargumentowe konstruktory (np. `Tree`) są rodzaju `* -> *`

    ~~~~ {.haskell}
    {-#LANGUAGE KindSignatures, ExplicitForAll #-}

    class Functor f => Pointed (f :: * -> *) where
        pure :: forall (a :: *).a -> f a
    ~~~~

* Występują też bardziej złożone rodzaje, np. dla transformatorów monad:

    ~~~~ {.haskell}
    class MonadTrans (t :: (* -> *) -> * -> *) where
        lift :: Monad (m :: * -> *) => forall (a :: *).m a -> t m a
    ~~~~

NB spacje są niezbędne - `::*->*` jest jednym leksemem.

# Klasy wieloparametrowe

* Czasami potrzebujemy opisać nie tyle pojedynczy typ, co relacje między typami:

    ~~~~ {.haskell}
    {-#LANGUAGE MultiParamTypeClasses, FlexibleInstances #-}
    class Iso a b where
      iso :: a -> b
      osi :: b -> a

    instance Iso a a where
      iso = id
      osi = id

    instance Iso ((a,b)->c) (a->b->c) where
      iso = curry
      osi = uncurry

    instance (Iso a b) => Iso [a] [b] where
     iso = map iso
     osi = map osi
    ~~~~

* Uwaga: w ostatnim przykładzie `iso` ma inny typ po lewej, inny po prawej

* Ćwiczenie: napisz jeszcze jakieś instancje klasy `Iso`


    ~~~~ {.haskell}
    instance (Functor f, Iso a b) => Iso (f a) (f b) where
    instance Iso (a->b->c) (b->a->c) where
    ~~~~

# Dygresja - FlexibleInstances

Haskell 2010

<!--
An instance declaration introduces an instance of a class. Let class
cx => C u where { cbody } be a class declaration. The general form of
the corresponding instance declaration is: instance cx′ => C (T u1 …
uk) where { d } where k ≥ 0. The type (T u1 … uk) must take the form
of a type constructor T applied to simple type variables u1, … uk;
furthermore, T must not be a type synonym, and the ui must all be
distinct.
-->

* an instance head must have the form C (T u1 ... uk), where T is a type constructor defined by a data or newtype declaration  and the ui are distinct type variables, and

<!--
*    each assertion in the context must have the form C' v, where v is one of the ui.
-->

This prohibits instance declarations such as:

  instance C (a,a) where ...
  instance C (Int,a) where ...
  instance C [[a]] where ...

`instance Iso a a` nie spełnia tych warunków, ale wiadomo o jaką relację nam chodzi :)

# Problem z klasami wieloparametrowymi
Spróbujmy stworzyć klasę kolekcji, np.

`BadCollection.hs`

~~~~ {.haskell}
class Collection c where
  insert :: e -> c -> c
  member :: e -> c -> Bool

instance Collection [a] where
     insert = (:)
     member = elem
~~~~

~~~~
    Couldn't match type `e' with `a'
      `e' is a rigid type variable bound by
          the type signature for member :: e -> [a] -> Bool
          at BadCollection.hs:7:6
      `a' is a rigid type variable bound by
          the instance declaration
          at BadCollection.hs:5:22
~~~~

Dlaczego?

# Problem z klasami wieloparametrowymi

~~~~ {.haskell}
class Collection c where
 insert :: e -> c -> c
 member :: e -> c -> Bool
~~~~

tłumaczy się (mniej więcej) do

~~~~
data ColDic c = CD
 {
 , insert :: forall e.e -> c -> c
 , member :: forall e.e -> c -> Bool
 }
~~~~

 ... nie o to nam chodziło.

~~~~ {.haskell}
instance Collection [a] where
   insert = (:)
   member = undefined
~~~~

~~~~
-- (:) :: forall t. t -> [t] -> [t]
ColList :: forall a. ColDic a
ColList = \@ a -> CD { insert = (:) @ a, member =
~~~~

# Problem z klasami wieloparametrowymi

 <!--- `BadCollection2.hs` -->
<!---
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleInstances #-}
-->

~~~~ {.haskell}
class Collection c e where
  insert :: e -> c -> c
  member :: e -> c -> Bool

instance Eq a => Collection [a] a where
  insert  = (:)
  member = elem

ins2 x y c = insert y (insert x c)
-- ins2 :: (Collection c e, Collection c e1) => e1 -> e -> c -> c

problem1 :: [Int]
problem1 = ins2 1 2 []
-- No instances for (Collection [Int] e0, Collection [Int] e1)
-- arising from a use of `ins2'

problem2 = ins2 'a' 'b' []
-- No instance for (Collection [a0] Char)
--       arising from a use of `ins2'

problem3 :: (Collection c0 Char, Collection c0 Bool) => c0 -> c0
problem3 = ins2 True 'a'
-- Tu problem akurat polega na tym, że to jest poprawne typowo
-- ...a chyba nie powinno być
~~~~

# Zależności funkcyjne
Czasami w klasach wieloparametrowych, jeden parametr wyznacza inny, np.

~~~~ {.haskell}
 class (Monad m) => MonadState s m | m -> s where ...

 class Collects e ce | ce -> e where
      empty  :: ce
      insert :: e -> ce -> ce
      member :: e -> ce -> Bool
~~~~

Problem: *Fundeps are very, very tricky.* - SPJ

Więcej: http://research.microsoft.com/en-us/um/people/simonpj/papers/fd-chr/

# Refleksja - czemu nie klasy konstruktorowe?

Problem kolekcji możemy rozwiązać np. tak:

~~~~ {.haskell}
class Collection c where
  insert :: e -> c e -> c e
  member :: Eq e => e -> c e-> Bool

instance Collection [] where
     insert x xs = x:xs
     member = elem
~~~~

ale nie rozwiązuje to problemu np. z monadą stanu:

~~~~ {.haskell}
 class (Monad m) => MonadState s m | m -> s where
   get :: m s
   put :: s -> m ()
~~~~

typ stanu `s` nie jest tu parametrem konstruktora `m`.

# Fundeps are very very tricky

~~~~ {.haskell}
class Mul a b c | a b -> c where
  (*) :: a -> b -> c

newtype Vec a = Vec [a]
instance Functor Vec where
  fmap f (Vec as) = Vec $ map f as

instance Mul a b c => Mul a (Vec b) (Vec c) where
  a * b = fmap (a*) b

f t x y = if t then  x * (Vec [y]) else y
~~~~

Jakiego typu jest f? Niech x::a, y::b.

Wtedy typem wyniku jest b i musimy mieć instancję `Mul a (Vec b) b`

Z kolei `a b -> c` implikuje, że `b = Vec c` dla pewnego c, czyli szukamy instancji

~~~~
Mul a (Vec (Vec c)) (Vec c)
~~~~

zastosowanie reguły `Mul a b c => Mul a (Vec b) (Vec c)` doprowadzi nas do `Mul a (Vec c) c`.

...i tak w kółko.


# Spróbujmy

~~~~ {.haskell}
Mul1.hs:16:21:
    Context reduction stack overflow; size = 21
    Use -fcontext-stack=N to increase stack size to N
      co :: c18 ~ Vec c19
      $dMul :: Mul a0 c17 c18
      $dMul :: Mul a0 c16 c17
      ...
      $dMul :: Mul a0 c1 c2
      $dMul :: Mul a0 c c1
      $dMul :: Mul a0 c0 c
      $dMul :: Mul a0 (Vec c0) c0
    When using functional dependencies to combine
      Mul a (Vec b) (Vec c),
        arising from the dependency `a b -> c'
        in the instance declaration at 3/Mul1.hs:13:10
      Mul a0 (Vec c18) c18,
        arising from a use of `mul' at 3/Mul1.hs:16:21-23
    In the expression: mul x (Vec [y])
    In the expression: if b then mul x (Vec [y]) else y
~~~~

(musimy użyć UndecidableInstances, żeby GHC w ogóle spróbowało - ten przykład pokazuje co jest 'Undecidable').

# Rodziny typów

Rodziny to funkcje na typach - jak na pierwszym wykładzie

~~~~ {.haskell}
{-# TypeFamilies #-}

data Zero = Zero
data Suc n = Suc n

type family m :+ n
type instance Zero :+ n = n
type instance (Suc m) :+ n = Suc(m:+n)

vhead :: Vec (Suc n) a -> a
vappend :: Vec m a -> Vec n a -> Vec (m:+n) a
~~~~

Trochę dalej powiemy sobie o nich bardziej systematycznie.
