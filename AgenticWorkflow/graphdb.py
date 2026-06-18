from mgraph_db.mgraph.MGraph import MGraph
from mgraph_db.mgraph.schemas.Schema__MGraph__Node import Schema__MGraph__Node
from mgraph_db.mgraph.schemas.Schema__MGraph__Node__Data import Schema__MGraph__Node__Data
from mgraph_db.mgraph.schemas.Schema__MGraph__Edge import Schema__MGraph__Edge, Schema__MGraph__Edge__Data
from mgraph_db.mgraph.schemas.Schema__MGraph__Graph import Schema__MGraph__Graph
from osbot_utils.type_safe.primitives.domains.identifiers.Edge_Id import Edge_Id
from osbot_utils.type_safe.primitives.domains.identifiers.Node_Id import Node_Id
from typing import Dict, List
from pydantic import BaseModel, Field
import threading


########################################################################
# Pydantic Objects
########################################################################

class NamedEntity(BaseModel):
    name: str = Field(description="Name of the entity extracted from the plot")
    description: str = Field(description="Description of the extracted entity")

class GraphTriplet(BaseModel):
    subject: NamedEntity = Field(description="The subject of the relationship")
    predicate: str = Field(description="The predicate of the relationship")
    object: NamedEntity = Field(description="The object of the relationship")



########################################################################
# MGraph Objects
########################################################################


# Custom Node for MGraph
class Custom_Node_Data(Schema__MGraph__Node__Data):
    name: str
    type: str
    description: str
    
class Custom_Node(Schema__MGraph__Node):
    node_data: Custom_Node_Data  # type: ignore
 

# Custom Edge for MGraph
class Custom_Edge_Data(Schema__MGraph__Edge__Data):
    predicate: str = ""  # This allows the 'predicate' key inside edge_data

class Custom_Edge(Schema__MGraph__Edge):
    edge_data: Custom_Edge_Data  # type: ignore


# Custom Graph Schema to preserve Custom_Node and Custom_Edge types when loading
class Custom_Graph(Schema__MGraph__Graph):
    nodes: Dict[Node_Id, Custom_Node]  # type: ignore
    edges: Dict[Edge_Id, Custom_Edge]  # type: ignore


class MGraphManager:
    """
    Manager class for handling MGraph operations.
    
    ## ONTOLOGY

    Entities:
    LONG TERM GOAL: One of SEARCH, HOUSING, FIRE_FIGHTING
    GOAL: Represents a subgoal of a long-term goal.
    GOAL_TYPE: One of RECON, EXTINGUISH, RESCUE
    POSITION: Represents a position in the graph.
    X-COORDINATE: A number representing X COORDINATE on the simulation map
    Y-COORDINATE: A number representing Y COORDINATE on the simulation map
    PRIORITY: One of HIGH, MEDIUM, LOW

    Relationships:
    - LONG TERM GOAL -- HAS_GOAL --> GOAL
    - GOAL -- HAS_TYPE --> GOAL_TYPE
    - GOAL -- HAS_POSITION --> POSITION
    - POSITION -- HAS_X_COORDINATE --> X-COORDINATE
    - POSITION -- HAS_Y_COORDINATE --> Y-COORDINATE
    - GOAL -- HAS_PRIORITY --> PRIORITY
    """
    def __init__(self):
        self.mgraph = MGraph()
        # Serialise all graph access: pydantic_ai runs tool calls concurrently
        # in a thread pool, and MGraph's underlying dicts are not thread-safe.
        self._lock = threading.Lock()
    
    def get_goals(self, long_term_goal: str) -> list[GraphTriplet]:
        """
        Get all goal entities associated with a long term goal

        Args:
            long_term_goal: The long term goal we are interested in.
        
        Returns:
            GraphTriplets starting with the long_term_goal as the subject and the goals as the object.
        """
        with self._lock:
            results = []
            target_node_id = None
            
            with self.mgraph.data() as data:
                for node in data.nodes():
                    if hasattr(node, 'node_data') and getattr(node.node_data, 'name', None) == long_term_goal:
                        target_node_id = node.node_id
                        break    
                
                if target_node_id is None:
                    return []

                for edge in data.edges():
                    if edge.from_node_id() == target_node_id or edge.to_node_id() == target_node_id:
                
                        subject = data.node(edge.from_node_id()).node_data.name
                        predicate = getattr(edge.edge.data.edge_data, 'predicate', None)
                        object = data.node(edge.to_node_id()).node_data.name

                        results.append(GraphTriplet(
                            subject=NamedEntity(name=subject, description=""),
                            predicate=predicate,
                            object=NamedEntity(name=object, description="")
                        ))

            return results

    def query_subgraph_by_entity_name(self, entity_name: str) -> list[GraphTriplet]:
        """
        Query the knowledge graph for a specific entity.
        
        Args:
            entity_name: The name of the entity to query for.
        
        Returns:
            GraphTriplets starting with the long_term_goal as the subject and the entity_name as the object.
        """ 
        with self._lock:
            results = []
            target_node_id = None

            with self.mgraph.data() as data:
                for node in data.nodes():
                    if hasattr(node, 'node_data') and getattr(node.node_data, 'name', None) == entity_name:
                        target_node_id = node.node_id
                        break    
            
                for edge in data.edges():
                    if edge.from_node_id() == target_node_id or edge.to_node_id() == target_node_id:
                
                        subject = data.node(edge.from_node_id()).node_data.name
                        predicate = getattr(edge.edge.data.edge_data, 'predicate', None)
                        object = data.node(edge.to_node_id()).node_data.name

                        results.append(GraphTriplet(
                            subject=NamedEntity(name=subject, description=""),
                            predicate=predicate,
                            object=NamedEntity(name=object, description="")
                        ))

            return results

    def query_subgraph_by_relationship_name(self, relationship_name: str) -> list[GraphTriplet]:
        """
        Query the knowledge graph for a specific relationship.
        
        Args:
            relationship_name: The name of the relationship to query for.
        
        Returns:
            GraphTriplets starting with the long_term_goal as the subject and the relationship_name as the object.
        """
        with self._lock:
            results = []

            with self.mgraph.data() as data:
                for edge in data.edges():
                    predicate = getattr(edge.edge.data.edge_data, 'predicate', None)
                    if predicate == relationship_name:
                        subject = data.node(edge.from_node_id()).node_data.name
                        object = data.node(edge.to_node_id()).node_data.name

                        results.append(GraphTriplet(
                            subject=NamedEntity(name=subject, description=""),
                            predicate=predicate,
                            object=NamedEntity(name=object, description="")
                        ))

            return results

    def _insert_or_get_node(self, node_name: str, node_type: str):
        """
        Insert a new node into the knowledge graph.
        
        Args:
            node_name: The name of the node to insert.
            node_type: The type of the node to insert.
        
        Returns:
            The node that was inserted or retrieved.
        NOTE: Caller must already hold self._lock.
        """
        with self.mgraph.data() as data:
            for node in data.nodes():
                if hasattr(node, 'node_data') and getattr(node.node_data, 'name', None) == node_name:
                    return node

        with self.mgraph.edit() as edit:
            node = edit.new_node(
                node_type=Custom_Node,
                name=node_name,
                type=node_type,
                description=""
            )
            return node

    def insert_goal(self, long_term_goal: str, goal: str, goal_type: str, position: str, position_x: str, position_y: str, priority: str) -> bool:
        """
        Insert a new goal into the knowledge graph.
        
        Args:
            long_term_goal: The long term goal we are interested in.
            goal: The name of the goal to insert.
            goal_type: The type of the goal.
            position: The position of the goal.
            position_x: The x coordinate of the goal.
            position_y: The y coordinate of the goal.
            priority: The priority of the goal.
        
        Returns:
            True if the goal was inserted successfully, False otherwise.
        """
        with self._lock:
            long_term_goal_node = self._insert_or_get_node(long_term_goal, 'LONG TERM GOAL')
            goal_node           = self._insert_or_get_node(goal, 'GOAL')
            goal_type_node      = self._insert_or_get_node(goal_type, 'GOAL_TYPE')
            position_node       = self._insert_or_get_node(position, 'POSITION')
            position_x_node     = self._insert_or_get_node(position_x, 'X-COORDINATE')
            position_y_node     = self._insert_or_get_node(position_y, 'Y-COORDINATE')
            priority_node       = self._insert_or_get_node(priority, 'PRIORITY')
            
            with self.mgraph.edit() as edit:
                edit.new_edge(from_node_id=long_term_goal_node.node_id, to_node_id=goal_node.node_id,
                              edge_type=Custom_Edge, edge_data={'predicate': "HAS_GOAL"})
                edit.new_edge(from_node_id=goal_node.node_id, to_node_id=goal_type_node.node_id,
                              edge_type=Custom_Edge, edge_data={'predicate': "HAS_GOAL_TYPE"})
                edit.new_edge(from_node_id=goal_node.node_id, to_node_id=position_node.node_id,
                              edge_type=Custom_Edge, edge_data={'predicate': "HAS_POSITION"})
                edit.new_edge(from_node_id=position_node.node_id, to_node_id=position_x_node.node_id,
                              edge_type=Custom_Edge, edge_data={'predicate': "HAS_X-COORDINATE"})
                edit.new_edge(from_node_id=position_node.node_id, to_node_id=position_y_node.node_id,
                              edge_type=Custom_Edge, edge_data={'predicate': "HAS_Y-COORDINATE"})
                edit.new_edge(from_node_id=goal_node.node_id, to_node_id=priority_node.node_id,
                              edge_type=Custom_Edge, edge_data={'predicate': "HAS_PRIORITY"})
            return True

    def delete_entity(self, entity_name: str) -> bool:
        """
        Delete an entity from the knowledge graph.
        
        Args:
            entity_name: The name of the entity to delete.
        
        Returns:
            True if the entity was deleted successfully, False otherwise.
        """
        with self._lock:
            target_node_id = None

            with self.mgraph.data() as data:
                for node in data.nodes():
                    if hasattr(node, 'node_data') and getattr(node.node_data, 'name', None) == entity_name:
                        target_node_id = node.node_id
                        break

            if target_node_id is None:
                return False

            with self.mgraph.edit() as edit:
                with self.mgraph.data() as data:
                    for edge in data.edges():
                        if edge.from_node_id() == target_node_id or edge.to_node_id() == target_node_id:
                            edit.delete_edge(edge.edge.edge_id)

                edit.delete_node(target_node_id)

            return True



