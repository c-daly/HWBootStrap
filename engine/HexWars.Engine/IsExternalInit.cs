// Compiler shim: enables C# 9 `record` types and `{ get; init; }` accessors when targeting
// netstandard2.1 (the Unity .NET Standard profile lacks System.Runtime.CompilerServices.IsExternalInit).
// Without this, records / init-only setters fail to compile with CS0518.
namespace System.Runtime.CompilerServices
{
    internal static class IsExternalInit { }
}
