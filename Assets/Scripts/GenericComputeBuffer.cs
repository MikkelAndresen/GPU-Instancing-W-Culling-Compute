using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using Unity.Collections;
using Unity.Profiling;
using UnityEngine;
using UnityEngine.Profiling;

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
	public delegate void BufferWriteDelegate<T>(ref NativeArray<T> matrices) where T : unmanaged;

	protected ComputeBuffer buffer;

	public abstract int Stride { get; }
	public abstract int Count { get; }
	public abstract ComputeBufferType BufferType { get; protected set; }
	public abstract ComputeBufferMode BufferMode { get; protected set; }
	public ComputeBuffer Buffer => buffer;

	protected bool EnsureBufferSize()
	{
		if (buffer == null || Count > buffer.count)
		{
			buffer?.Dispose();
			if (Count > 0)
			{
				// Double buffer size whenever we exceed capacity
				buffer = new ComputeBuffer(Count * 2, Stride, BufferType, BufferMode);
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
				SetBufferData(-1);
			else
				SetBufferData(subset);
		}
		return reallocated;
	}

	private ProfilerMarker beginWriteMatricesMarker = new ProfilerMarker(nameof(BeginWriteMatrices));
	/// <summary>
	/// Uses <see cref="ComputeBuffer.BeginWrite{T}(int, int)"/>
	/// </summary>
	/// <typeparam name="T"></typeparam>
	/// <param name="startIndex"></param>
	/// <param name="writeCount"></param>
	/// <param name="del"></param>
	/// <returns>An action which runs <see cref="ComputeBuffer.EndWrite{T}(int)"/></returns>
	public Action BeginWriteMatrices<T>(int startIndex, int writeCount, BufferWriteDelegate<T> del) where T : unmanaged
	{
		if (writeCount > Count || writeCount <= 0)
			throw new ArgumentOutOfRangeException($"{nameof(writeCount)} cannot <= 0 or > than {Count}");

		beginWriteMatricesMarker.Begin();

		EnsureBufferSize();
		var array = Buffer.BeginWrite<T>(0, writeCount);

		del(ref array);

		beginWriteMatricesMarker.End();
		return ()=> Buffer.EndWrite<T>(writeCount);
	}

	protected abstract void SetBufferData(DataSubset subset);
	protected abstract void SetBufferData(int index, int count = -1);

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
	public override ComputeBufferMode BufferMode { get; protected set; }
	public GenericComputeBuffer(
		IList<T> data, 
		ComputeBufferType bufferType = ComputeBufferType.Default, 
		ComputeBufferMode mode = default)
	{
		this.data = data;
		this.BufferType = bufferType;
		this.BufferMode = mode;
	}

	protected override void SetBufferData(DataSubset subset)
	{
		if (subset == null)
			throw new ArgumentNullException(nameof(subset));

		SetBufferData(subset.startIndex, subset.count);
	}

	protected override void SetBufferData(int index, int count = 1)
	{
		if (data is T[] array)
		{
			if (index > -1)
				buffer.SetData(array, index, index, count);
			else
				buffer.SetData(array);
		}
		else if (data is List<T> list)
		{
			if (index > -1)
				buffer.SetData(list, index, index, count);
			else
				buffer.SetData(list);
		}
		else
			Debug.LogError("DataBuffer unhandled collection type " + data.GetType());
	}
}

public class GenericNativeComputeBuffer<T> : AbstractComputeBuffer where T : struct
{
	private NativeArray<T> data;

	public override int Count => data.Length;
	public override int Stride => Marshal.SizeOf<T>();
	public override ComputeBufferType BufferType { get; protected set; }
	public override ComputeBufferMode BufferMode { get; protected set; }

	public GenericNativeComputeBuffer(
		NativeArray<T> data, 
		ComputeBufferType bufferType = ComputeBufferType.Default, 
		ComputeBufferMode mode = default)
	{
		this.data = data;
		this.BufferType = bufferType;
		this.BufferMode = mode;
	}

	protected override void SetBufferData(DataSubset subset)
	{
		if (subset == null)
			throw new ArgumentNullException(nameof(subset));

		SetBufferData(subset.startIndex, subset.count);
	}

	protected override void SetBufferData(int index, int count = 1)
	{
		if (index > -1)
			buffer.SetData(data, index, index, count);
		else
			buffer.SetData(data);
	}
}