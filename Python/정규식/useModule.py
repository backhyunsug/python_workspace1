import module1 

print( module1.add(4,5) )
print( module1.sub(4,5) )

import module1 as mod2 

print( mod2.add(4,5) )
print( mod2.sub(4,5) )

from module1 import add, sub
print( add(4,5) )
print( sub(4,5) )

from module1 import Test
t1 = Test(6,7)
t1.display()