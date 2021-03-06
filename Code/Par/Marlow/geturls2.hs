import GetURL

import Control.Monad
import Control.Concurrent
data Async a = Async (MVar a)

async :: IO a -> IO (Async a)
async action = do
   var <- newEmptyMVar
   forkIO (action >>= putMVar var)
   return (Async var)

wait :: Async a -> IO a
wait (Async var) = readMVar var

main = do
  m1 <- async $ getURL "http://evemaps.dotlan.net/map/Domain"
  m2 <- async $ getURL "http://evemaps.dotlan.net/map/Lonetrek"
  wait m1
  print "1 DONE"
  wait m2
  print "2 DONE"
