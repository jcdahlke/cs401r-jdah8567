output "feature_group_name" {
  description = "Name of the customer Feature Group"
  value       = aws_sagemaker_feature_group.customer.feature_group_name
}

output "feature_group_arn" {
  description = "ARN of the customer Feature Group"
  value       = aws_sagemaker_feature_group.customer.arn
}
