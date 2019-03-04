# Stack

1.  Na własnym komputerze
    * Zainstalować `stack`
    * Zainstalować GHC 7.10 przy pomocy `stack setup`
    * Zainstalować GHC 8 przy pomocy `stack setup`
    * Uruchomić `stack ghci` 7.10 i 8
    * Zbudować i uruchomić projekt hello, trochę go zmodyfikować
    * `stack install QuickCheck`
    
2. Na students
   * Można zrobić to samo co powyżej, ale możliwy problem z quota
   * `stack setup` przy użyciu GHC 7.10 (i może 7.8) z PUBLIC
   * Reszta jak wyżej

Jeżeli ktoś nie jest w stanie użyć `stack`, można `cabal install doctest HUnit QuickCheck`

# QuickCheck
* Wypróbuj testy przy uzyciu `doctest` i `HUnit`
* Napisz kilka testów dla sortowania [Int]
* Napisz kilka testów sprawdzających własności funkcji na listach (++,concat,map,etc.)
* Napisz generator dla OrderedInts (posortowanych [Int])
* Napisz generator dla jakichś drzew (np. BST) i użyj go w testach
* Bazując na notatkach z wykładu, odtwórz QuickCheck v1 (i ulepsz go)
