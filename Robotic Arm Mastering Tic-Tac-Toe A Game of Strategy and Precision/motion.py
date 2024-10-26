
from pyniryo2 import *
import pyniryo
from matplotlib import pyplot as plt
import cv2 as cv
import time
from recorded_positions import pickup_coordinates,dropoff_coordinates

robot_ip_address = "169.254.200.200"
robot = NiryoRobot(robot_ip_address)
def get_pickup_coordinates(p):
    return pickup_coordinates.get(p,'Invalid_position')
def get_dropoff_coordinates(p):
    return dropoff_coordinates.get(p,'Invalid_position')

def go_up(position_coordinates):
    coordinates=[0,0,0.03 ,0,0,0]
    result = [a + b for a, b in zip(position_coordinates, coordinates)]
    robot.arm.move_pose(result)

def go_down(position_coordinates):
    coordinates=[0,0,0.03 ,0,0,0]
    result = [a + b for a, b in zip(position_coordinates, coordinates)]
    robot.arm.move_pose(result)



def change_indication():
    go_up(get_pickup_coordinates('i2'))
    robot.arm.move_pose(get_pickup_coordinates('i2'))
    robot.tool.grasp_with_tool()
    go_up(get_pickup_coordinates('i2'))

    go_down(get_dropoff_coordinates('i1'))
    robot.arm.move_pose(get_dropoff_coordinates('i1'))
    robot.tool.release_with_tool()

    robot.arm.move_pose(get_dropoff_coordinates('h0'))




def movement(positions_robot, pickup_index, positions_player=0):
    home_position = get_pickup_coordinates('h0')
    robot.arm.move_pose(home_position)

    pick_up_list = ['c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9', 'c10']

    if pickup_index <= 4:
            # Move to the pickup location
        robot.arm.move_pose(get_pickup_coordinates(pick_up_list[pickup_index]))
        robot.tool.grasp_with_tool()
            
            # Go up (assuming you have a function to lift the item)
        go_up(get_pickup_coordinates(pick_up_list[pickup_index]))

            # Move to the drop-off location
        go_down(get_dropoff_coordinates(f'p{positions_robot}'))
        robot.arm.move_pose(get_dropoff_coordinates(f'p{positions_robot}'))
            
            # Release the tool
        robot.tool.release_with_tool()
        #robot.arm.move_pose(get_dropoff_coordinates('h0'))

            # Move back to the home position
        change_indication()

              
        

def clear_out(robot_positions_stored, player_positions_stored,r):
        pick_up_list_1=['c5','c4','c3','c2','c1']
        pick_up_list_2=['c10','c9','c8','c7','c16']
        if r==2:
        	return None

        for i in range(len(robot_positions_stored)):
            print('for loop 1',i)
            robot.arm.move_pose(get_pickup_coordinates(robot_positions_stored[i]))
            robot.tool.grasp_with_tool()
            go_up(get_pickup_coordinates(robot_positions_stored[i]))

            go_down(get_dropoff_coordinates(pick_up_list_1[i]))

            robot.arm.move_pose(get_dropoff_coordinates(pick_up_list_1[i]))
            robot.tool.release_with_tool()

            robot.arm.move_pose(get_dropoff_coordinates('h0'))
            print('for loop end')

        print(player_positions_stored)
        print(robot_positions_stored)
        
        for i in range(len(player_positions_stored)):
            print('for loop 1',i)
            robot.arm.move_pose(get_pickup_coordinates( player_positions_stored[i]))
            robot.tool.grasp_with_tool()
            go_up(get_pickup_coordinates( player_positions_stored[i]))

            go_down(get_dropoff_coordinates(pick_up_list_2[i]))

            robot.arm.move_pose(get_dropoff_coordinates(pick_up_list_2[i]))
            robot.tool.release_with_tool()

            robot.arm.move_pose(get_dropoff_coordinates('h0'))
        
