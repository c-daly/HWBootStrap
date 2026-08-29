using System;
using System.Globalization;
using System.IO;
using Newtonsoft.Json;

namespace HexWars.Presentation
{
    public static class PolicyJson
    {
        static readonly JsonSerializerSettings Settings = new JsonSerializerSettings
        {
            NullValueHandling = NullValueHandling.Include,
            Culture = CultureInfo.InvariantCulture,
            Formatting = Formatting.None,
            Converters = { new FiniteNumberConverter() },
        };

        public static string Serialize(object value)
        {
            if (value == null) throw new ArgumentNullException(nameof(value));
            using var buffer = new StringWriter(CultureInfo.InvariantCulture);
            using var writer = new JsonTextWriter(buffer)
            {
                Culture = CultureInfo.InvariantCulture,
                Formatting = Formatting.None,
            };
            JsonSerializer.CreateDefault(Settings).Serialize(writer, value);
            return buffer.ToString();
        }

        sealed class FiniteNumberConverter : JsonConverter
        {
            public override bool CanRead => false;

            public override bool CanConvert(Type objectType) =>
                objectType == typeof(float) || objectType == typeof(double);

            public override void WriteJson(
                JsonWriter writer, object value, JsonSerializer serializer)
            {
                if (value is float single)
                {
                    if (float.IsNaN(single) || float.IsInfinity(single))
                        throw new JsonSerializationException(
                            "policy JSON does not permit non-finite floating-point values");
                    writer.WriteValue(single);
                    return;
                }

                double number = (double)value;
                if (double.IsNaN(number) || double.IsInfinity(number))
                    throw new JsonSerializationException(
                        "policy JSON does not permit non-finite floating-point values");
                writer.WriteValue(number);
            }

            public override object ReadJson(
                JsonReader reader, Type objectType, object existingValue,
                JsonSerializer serializer) => throw new NotSupportedException();
        }
    }
}
