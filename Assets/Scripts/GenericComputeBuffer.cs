using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using UnityEngine;

public class DataSubset
{
	public int startIndex;
	public int count;

	public DataSubset(int startIndex, int count)
	{
		this.startIndex = startIndex;
		this.count = count;
	}
}

public abstract class AbstractComputeBuffer : IDisposable
{
	protected ComputeBuffer buffer;

	public abstract int Stride { get; }
	public abstract int Count { get; }
	public abstract ComputeBufferType BufferType { get; protected set; }
	public ComputeBuffer Buffer => buffer;

	protected bool EnsureBufferSize()
	{
		if (buffer == null || Count > buffer.count)
		{
			buffer?.Dispose();
			if (Count > 0)
			{
				// Double buffer size whenever we exceed capacity
				buffer = new ComputeBuffer(Count * 2, Stride, BufferType);
			}
			else
				buffer = null;
			return true;
		}
		return false;
	}

	public bool SetData(int index)
	{
		bool reallocated = EnsureBufferSize();
		if (buffer != null)
		{
			if (reallocated)
				SetBufferData(-1);
			else
				SetBufferData(index);
		}
		return reallocated;
	}

	public bool SetData(DataSubset subset)
	{
		bool reallocated = EnsureBufferSize();
		if (buffer != null)
		{
			if (reallocated)
				SetBufferData(null);
			else
				SetBufferData(subset);
		}
		return reallocated;
	}

	protected abstract void SetBufferData(DataSubset subset);
	protected abstract void SetBufferData(int index);

	public void Dispose() 
	{
		if (buffer != null)
		{
			buffer.Release();
			buffer = null;
		}
	}
}

public class GenericComputeBuffer<T> : AbstractComputeBuffer where T : struct
{
	private IList<T> data;

	public override int Count => data.Count;
	public override int Stride => Marshal.SizeOf<T>();
	public override ComputeBufferType BufferType { get; protected set; }

	public GenericComputeBuffer(IList<T> data, ComputeBufferType bufferType = ComputeBufferType.Default)
	{
		this.data = data;
		this.BufferType = bufferType;
	}

	protected override void SetBufferData(DataSubset subset)
	{
		if (data is T[] array)
		{
			if (subset != null)
				buffer.SetData(array, subset.startIndex, subset.startIndex, subset.count);
			else
				buffer.SetData(array);
		}
		else if (data is List<T> list)
		{
			if (subset != null)
				buffer.SetData(list, subset.startIndex, subset.startIndex, subset.count);
			else
				buffer.SetData(list);
		}
		else
			Debug.LogError("DataBuffer unhandled collection type " + data.GetType());
	}

	protected override void SetBufferData(int index)
	{
		if (data is T[] array)
		{
			if (index > -1)
				buffer.SetData(array, index, index, 1);
			else
				buffer.SetData(array);
		}
		else if (data is List<T> list)
		{
			if (index > -1)
				buffer.SetData(list, index, index, 1);
			else
				buffer.SetData(list);
		}
	}
}