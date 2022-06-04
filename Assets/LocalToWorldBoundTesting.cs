using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class LocalToWorldBoundTesting : MonoBehaviour
{
	[SerializeField]
	private Bounds localBounds;
	[SerializeField]
	private Bounds worldBounds;

	private void OnDrawGizmos()
	{
		Gizmos.color = Color.green;
		Vector3 origin = transform.position;
		Vector3 extents = localBounds.extents;
		Quaternion orientation = transform.rotation;
		Vector3 max = localBounds.max;
		Vector3 min = localBounds.min;
		//Vector2 sizeX = origin.x + (x - origin.x) * Mathf.Cos(theta) + (y - origin.y) * Mathf.Sin(theta);
		//Vector2 sizeY = origin.y - (x - origin.y) * Mathf.Sin(theta) + (y - origin.y) * Mathf.Cos(theta);

		//Vector3 extents = transform.forward * localBounds.extents.z + transform.right * localBounds.extents.x + transform.up * localBounds.extents.y;
		worldBounds.max = origin + new Vector3(Mathf.Sin(orientation.x), 0, Mathf.Cos(orientation.y)) + extents;
		worldBounds.min = origin + new Vector3(Mathf.Cos(orientation.x), 0, Mathf.Sin(orientation.y)) + extents;
		Gizmos.DrawWireCube(transform.position, worldBounds.extents);
	}
}