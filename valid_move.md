# A move is valid if : 

### The target square is valid and available : 
- If the move coordinates are not out of bound ✅ _method_ : `is_valid_coord` (was already coded in the sample code)
- If the move coordinates are adjacent to the current square or on the current ✅ _method_ : `is_dst_valid_square`
### If the moving entity is allowed to move : 
- If the moving entity is an AI, a Firewall or a Program :
- - it cannot move if it's adjacent to an enemy ✅ _method_ : `is_moving_unit_allowed_to_move`
- - It can only move right or down if it's a defending entity ✅ _method_ : `is_dst_valid_square`
- - It can only move left or up if it's an attacking entity ✅ _method_ : `is_dst_valid_square`
### The moving entity is allowed to repair : 
- A tech can repair Firewall or Program ✅ _method_ : `is_unit_allowed_to_repair_dst`
- An AI can repair virus and tech, ✅ _method_ : `is_unit_allowed_to_repair_dst`
- The health of the repairee must not be full ✅ _method_ : `is_unit_allowed_to_repair_dst`


